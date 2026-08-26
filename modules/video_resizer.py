"""
Video downscale via ffmpeg (Tahap 1).

Hanya menurunkan resolusi + kompres ulang; tidak ada crop/pad/preset sosmed.
Rasio aspek selalu dipertahankan. Semua kerja berat dilakukan oleh binary
ffmpeg (C/hardware), jadi tidak ada loop per-frame di Python.
"""
import json
import re
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

# Binary dicari sekali saat import. Di Docker keduanya ada (apt install ffmpeg);
# di build .exe (PyInstaller) keduanya None -> tab video menampilkan pesan.
FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")

ALLOWED_EXT = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}

# Target = sisi pendek frame (1080 untuk landscape 1920x1080 maupun portrait
# 1080x1920), mengikuti cara orang menyebut "1080p".
TARGET_HEIGHTS = {"1080": 1080, "720": 720, "480": 480}
QUALITY_CRF = {"high": 20, "medium": 23, "low": 28}
DEFAULT_TARGET = "1080"
DEFAULT_QUALITY = "medium"

# Satu encode aktif pada satu waktu. ffmpeg sudah memakai semua core; dua job
# paralel hanya membuat keduanya lambat dan rawan OOM.
_ENCODE_SLOT = threading.Semaphore(1)

_TIME_RE = re.compile(r"(\d+):(\d{2}):(\d{2}(?:\.\d+)?)")


def is_available() -> bool:
    return bool(FFMPEG and FFPROBE)


def unavailable_reason() -> str:
    if not FFMPEG:
        return "ffmpeg tidak ditemukan di sistem."
    if not FFPROBE:
        return "ffprobe tidak ditemukan di sistem."
    return ""


def _run(cmd: list, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def probe(path: Path) -> dict:
    """Baca metadata video. Return {status, width, height, duration, has_audio, ...}."""
    if not FFPROBE:
        return {"status": "error", "error": unavailable_reason()}
    cmd = [FFPROBE, "-v", "error", "-print_format", "json",
           "-show_streams", "-show_format", str(path)]
    try:
        proc = _run(cmd)
    except subprocess.TimeoutExpired:
        return {"status": "error", "error": "ffprobe timeout - file mungkin rusak."}
    if proc.returncode != 0:
        return {"status": "error", "error": "File bukan video valid atau codec tidak dikenali."}

    try:
        info = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"status": "error", "error": "Gagal membaca metadata video."}

    vstream = next((s for s in info.get("streams", [])
                    if s.get("codec_type") == "video"), None)
    if not vstream:
        return {"status": "error", "error": "Tidak ada stream video di file ini."}

    width = int(vstream.get("width") or 0)
    height = int(vstream.get("height") or 0)
    if width <= 0 or height <= 0:
        return {"status": "error", "error": "Dimensi video tidak terbaca."}

    # Video dari HP sering disimpan landscape + metadata rotate 90 derajat.
    # ffmpeg menerapkan display matrix saat decode, jadi filter melihat dimensi
    # yang sudah diputar - probe harus ikut menukar agar keputusan skala benar.
    if _rotation_of(vstream) in (90, 270):
        width, height = height, width

    duration = 0.0
    for src in (vstream.get("duration"), info.get("format", {}).get("duration")):
        try:
            duration = float(src)
            if duration > 0:
                break
        except (TypeError, ValueError):
            continue

    astream = next((s for s in info.get("streams", [])
                    if s.get("codec_type") == "audio"), None)

    return {
        "status": "ok",
        "width": width,
        "height": height,
        "duration": duration,
        "has_audio": astream is not None,
        "vcodec": vstream.get("codec_name", "?"),
        "acodec": (astream or {}).get("codec_name", ""),
    }


def _rotation_of(vstream: dict) -> int:
    """Ambil rotasi dari side_data (ffmpeg baru) atau tag rotate (lama)."""
    for sd in vstream.get("side_data_list", []) or []:
        if "rotation" in sd:
            try:
                return abs(int(float(sd["rotation"]))) % 360
            except (TypeError, ValueError):
                pass
    try:
        return abs(int(float(vstream.get("tags", {}).get("rotate", 0)))) % 360
    except (TypeError, ValueError):
        return 0


def _scale_filter(width: int, height: int, target: int) -> str:
    """Kunci sisi pendek ke target, sisi lain ikut rasio (-2 = kelipatan 2)."""
    if width >= height:          # landscape / persegi -> tinggi yang dikunci
        return f"scale=-2:{target}:flags=lanczos"
    return f"scale={target}:-2:flags=lanczos"     # portrait -> lebar yang dikunci


def _parse_time(value: str) -> float:
    m = _TIME_RE.match(value.strip())
    if not m:
        return 0.0
    h, mnt, sec = m.groups()
    return int(h) * 3600 + int(mnt) * 60 + float(sec)


def _build_cmd(in_path: Path, out_path: Path, vf: str, crf: int,
               audio_mode: str) -> list:
    # -loglevel error: kita hanya membaca stderr setelah proses selesai, jadi
    # stderr yang bertele-tele (banner + dump stream) berisiko memenuhi buffer
    # pipe dan membuat ffmpeg menggantung. Batasi ke pesan error saja — itu
    # juga yang dipakai untuk pesan kegagalan ke user.
    cmd = [
        FFMPEG, "-nostdin", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(in_path),
        "-vf", vf,
        "-c:v", "libx264", "-crf", str(crf), "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
    ]
    if audio_mode == "none":
        cmd += ["-an"]
    elif audio_mode == "aac":
        cmd += ["-c:a", "aac", "-b:a", "128k"]
    else:
        cmd += ["-c:a", "copy"]
    cmd += ["-movflags", "+faststart", "-progress", "pipe:1", "-nostats",
            str(out_path)]
    return cmd


def _encode(cmd: list, duration: float, progress_cb, cancel_cb) -> tuple:
    """Jalankan ffmpeg, streaming progress. Return (returncode, tail_stderr).

    stderr diarahkan ke file sementara, BUKAN pipe. Kita hanya membaca stderr
    setelah proses selesai, sementara buffer pipe terbatas (4-64KB): ffmpeg
    yang cerewet akan memblokir tulisannya sendiri sambil kita menunggu stdout
    -> deadlock. File sementara tidak punya batas itu.
    """
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8",
                                errors="replace") as errf:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errf,
                                text=True, bufsize=1)
        cancelled = False
        try:
            for line in proc.stdout:
                if cancel_cb and cancel_cb():
                    cancelled = True
                    break
                if not progress_cb or duration <= 0:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                elapsed = None
                if key == "out_time":
                    elapsed = _parse_time(value)
                elif key == "out_time_us":
                    try:
                        elapsed = float(value) / 1_000_000
                    except ValueError:
                        elapsed = None
                if elapsed is not None:
                    progress_cb(max(0.0, min(99.0, elapsed / duration * 100)))
            if not cancelled:
                proc.wait()
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=10)
            if proc.stdout:
                proc.stdout.close()
        if cancelled:
            return -1, "dibatalkan"
        errf.seek(0)
        stderr = errf.read()
    return proc.returncode, stderr[-800:]


def process_video(in_path: Path, out_dir: Path, target: str = DEFAULT_TARGET,
                  quality: str = DEFAULT_QUALITY, original_name: str = None,
                  progress_cb=None, cancel_cb=None) -> dict:
    """
    Turunkan resolusi satu video. Return {status, output_id, ...}.
    status: ok | skipped | error
    """
    if not is_available():
        return {"status": "error", "error": unavailable_reason()}

    target_px = TARGET_HEIGHTS.get(str(target), TARGET_HEIGHTS[DEFAULT_TARGET])
    crf = QUALITY_CRF.get(quality, QUALITY_CRF[DEFAULT_QUALITY])

    meta = probe(in_path)
    if meta["status"] != "ok":
        return meta

    src_w, src_h = meta["width"], meta["height"]
    if min(src_w, src_h) <= target_px:
        # Menaikkan resolusi hanya memperbesar file tanpa menambah detail.
        return {
            "status": "skipped",
            "error": f"Video sudah {src_w}x{src_h} - tidak lebih besar dari target {target_px}p.",
            "orig_res": f"{src_w}x{src_h}",
        }

    stem = Path(original_name).stem if original_name else in_path.stem
    out_name = f"{in_path.stem}_{target_px}p.mp4"
    out_path = out_dir / out_name
    vf = _scale_filter(src_w, src_h, target_px)
    duration = meta["duration"]

    with _ENCODE_SLOT:
        audio_mode = "copy" if meta["has_audio"] else "none"
        rc, err = _encode(_build_cmd(in_path, out_path, vf, crf, audio_mode),
                          duration, progress_cb, cancel_cb)

        # Sumber .mkv/.webm sering memakai codec audio yang tidak sah di MP4
        # (vorbis, pcm). Copy gagal -> encode ulang audio ke AAC.
        if rc != 0 and audio_mode == "copy" and not (cancel_cb and cancel_cb()):
            out_path.unlink(missing_ok=True)
            rc, err = _encode(_build_cmd(in_path, out_path, vf, crf, "aac"),
                              duration, progress_cb, cancel_cb)

    if rc != 0:
        out_path.unlink(missing_ok=True)   # jangan tinggalkan file parsial
        if cancel_cb and cancel_cb():
            return {"status": "error", "error": "Dibatalkan."}
        return {"status": "error", "error": f"ffmpeg gagal: {err.strip()[-200:]}"}

    if not out_path.exists() or out_path.stat().st_size == 0:
        out_path.unlink(missing_ok=True)
        return {"status": "error", "error": "ffmpeg selesai tapi output kosong."}

    out_meta = probe(out_path)
    new_w = out_meta.get("width", 0) if out_meta["status"] == "ok" else 0
    new_h = out_meta.get("height", 0) if out_meta["status"] == "ok" else 0

    in_mb = round(in_path.stat().st_size / 1024 ** 2, 1)
    out_mb = round(out_path.stat().st_size / 1024 ** 2, 1)

    in_path.unlink(missing_ok=True)

    return {
        "status": "ok",
        "output_id": out_name,
        "stem": stem,
        "orig_res": f"{src_w}x{src_h}",
        "new_res": f"{new_w}x{new_h}" if new_w else f"{target_px}p",
        "duration": round(duration, 1),
        "size_mb_before": in_mb,
        "size_mb_after": out_mb,
        "saved_pct": round((1 - out_mb / in_mb) * 100) if in_mb > 0 else 0,
    }
