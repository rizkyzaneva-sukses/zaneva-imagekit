import os
import sys
import uuid
import shutil
import threading
import time
import tempfile
import zipfile
from datetime import timedelta
from io import BytesIO
from pathlib import Path
from functools import wraps

from flask import (
    Flask, request, session, redirect, url_for,
    render_template, send_file, jsonify, abort
)
from dotenv import load_dotenv

# Enable HEIC/HEIF support (foto iPhone) untuk Pillow.
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    _HEIC_OK = True
except Exception as e:  # pragma: no cover
    print(f"[ImageKit] HEIC support tidak aktif: {e}")
    _HEIC_OK = False

# Versi exe (PyInstaller): baca .env yang diletakkan di samping exe-nya,
# apa pun working directory saat app dijalankan.
if getattr(sys, "frozen", False):
    load_dotenv(Path(sys.executable).parent / ".env")
load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "changeme-imagekit")
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(hours=12)
app.config["SESSION_PERMANENT"] = True

APP_PASSWORD = os.environ.get("APP_PASSWORD", "zaneva2024")
MAX_FILES = int(os.environ.get("MAX_FILES", 30))
MAX_FILE_MB = int(os.environ.get("MAX_FILE_SIZE_MB", 20))
# Cross-platform temp dir. On Linux/Docker -> /tmp/imagekit, on Windows -> %TEMP%\imagekit.
# Override with TMP_DIR env var if needed.
TMP_BASE = Path(os.environ.get("TMP_DIR") or (Path(tempfile.gettempdir()) / "imagekit"))
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp"}
if _HEIC_OK:
    ALLOWED_EXT |= {".heic", ".heif"}
MAX_VIDEO_MB = int(os.environ.get("MAX_VIDEO_SIZE_MB", 1500))
MAX_VIDEO_FILES = int(os.environ.get("MAX_VIDEO_FILES", 3))
APP_VERSION = "1.1.1"

TMP_BASE.mkdir(parents=True, exist_ok=True)

# ─── Import modules ───
from modules import bg_remover, upscaler, resizer, retoucher, video_resizer

# Aturan upload per tab. Tab gambar tetap memakai batas lama persis; video
# punya slot sendiri supaya menaikkan limit video tidak ikut melonggarkan
# tab foto (dulu ALLOWED_EXT/MAX_FILE_MB dipakai bersama semua tab).
_IMG_RULE = {"ext": ALLOWED_EXT, "max_mb": MAX_FILE_MB, "max_files": MAX_FILES,
             "reject": "Format tidak didukung (JPG/PNG/WEBP)"}
TAB_UPLOAD = {
    "bg": _IMG_RULE,
    "upscale": _IMG_RULE,
    "resize": _IMG_RULE,
    "retouch": _IMG_RULE,
    "video": {"ext": video_resizer.ALLOWED_EXT, "max_mb": MAX_VIDEO_MB,
              "max_files": MAX_VIDEO_FILES,
              "reject": "Format tidak didukung (MP4/MOV/MKV/WEBM)"},
}

# Semua model lazy-load: BG remover dimuat saat pertama dipakai (~beberapa
# detik untuk isnet), upscaler saat tab Upscale dibuka. Startup jadi instan.
print("[ImageKit] Startup ready. Models load lazily on first use.")


# ─── Auto-cleanup (24h) ───
def auto_cleanup():
    while True:
        try:
            now = time.time()
            for d in TMP_BASE.iterdir():
                if d.is_dir() and (now - d.stat().st_mtime) > 86400:
                    shutil.rmtree(d, ignore_errors=True)
                    print(f"[Cleanup] Removed: {d.name}")
        except Exception as e:
            print(f"[Cleanup] Error: {e}")
        time.sleep(3600)

threading.Thread(target=auto_cleanup, daemon=True).start()


# ─── Helpers ───
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("authenticated"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


def get_work_dir(tab: str) -> Path:
    """Get/create session work dir for a specific tab."""
    sid = session.get("sid")
    if not sid:
        sid = uuid.uuid4().hex
        session["sid"] = sid
    work = TMP_BASE / sid / tab
    (work / "input").mkdir(parents=True, exist_ok=True)
    (work / "output").mkdir(parents=True, exist_ok=True)
    return work


def _check_disk(need_bytes: int = 0):
    """Cek sisa disk. need_bytes = perkiraan ruang yang akan dipakai job ini.

    Untuk foto (need_bytes=0) perilakunya sama seperti sebelumnya: minimal 1GB
    bebas. Untuk video, threshold flat tidak cukup — satu job bisa menulis
    ratusan MB sampai GB, jadi kebutuhannya dihitung di depan.
    """
    stat = shutil.disk_usage(str(TMP_BASE))
    required = need_bytes + (1024 ** 3)   # selalu sisakan 1GB headroom
    if stat.free < required:
        return False, (f"Disk tidak cukup: butuh ~{required / 1024**3:.1f}GB, "
                       f"tersisa {stat.free / 1024**3:.1f}GB")
    return True, None


def handle_upload(tab: str):
    """Common upload handler for all tabs."""
    # Body multipart ditulis penuh ke disk, jadi ukurannya ikut diperhitungkan.
    # Penting untuk video: 1GB headroom saja tidak cukup menampung upload 1.5GB.
    ok, err = _check_disk(request.content_length or 0)
    if not ok:
        return jsonify({"error": err}), 507

    files = request.files.getlist("photos")
    if not files:
        return jsonify({"error": "Tidak ada file yang dikirim."}), 400

    cfg = TAB_UPLOAD.get(tab, _IMG_RULE)
    work = get_work_dir(tab)
    accepted, rejected = [], []

    for f in files[:cfg["max_files"]]:
        ext = Path(f.filename).suffix.lower()
        if ext not in cfg["ext"]:
            rejected.append({"name": f.filename, "reason": cfg["reject"]})
            continue
            
        safe_name = f"{uuid.uuid4().hex}{ext}"
        out_path = work / "input" / safe_name
        
        # Read in chunks to prevent memory exhaustion (DoS)
        saved_size = 0
        is_too_large = False
        with open(out_path, "wb") as out_file:
            while True:
                chunk = f.stream.read(8192)
                if not chunk:
                    break
                saved_size += len(chunk)
                if saved_size > cfg["max_mb"] * 1024 * 1024:
                    is_too_large = True
                    break
                out_file.write(chunk)

        if is_too_large:
            out_path.unlink() # Hapus file yang terpotong
            rejected.append({"name": f.filename, "reason": f"Ukuran melebihi {cfg['max_mb']}MB"})
            continue

        accepted.append({"id": safe_name, "original": f.filename})

    if len(files) > cfg["max_files"]:
        rejected.append({"name": "...", "reason": f"Hanya {cfg['max_files']} file pertama yang diproses"})

    return jsonify({"accepted": accepted, "rejected": rejected})


# ─── Auth ───
@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        if request.form.get("password", "").strip() == APP_PASSWORD:
            session.permanent = True
            session["authenticated"] = True
            session.pop("sid", None)
            return redirect(url_for("index"))
        error = "Password salah. Coba lagi."
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    sid = session.get("sid")
    if sid:
        shutil.rmtree(TMP_BASE / sid, ignore_errors=True)
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
@login_required
def index():
    return render_template("index.html",
                           max_files=MAX_FILES,
                           max_mb=MAX_FILE_MB,
                           max_video_files=MAX_VIDEO_FILES,
                           max_video_mb=MAX_VIDEO_MB,
                           video_ok=video_resizer.is_available(),
                           video_reason=video_resizer.unavailable_reason(),
                           version=APP_VERSION,
                           presets=resizer.PLATFORM_PRESETS)


# ─── Status ───
@app.route("/status")
def status():
    disk = shutil.disk_usage(str(TMP_BASE))
    tmp_size = sum(f.stat().st_size for f in TMP_BASE.rglob("*") if f.is_file())
    return jsonify({
        "version": APP_VERSION,
        "models": {
            "rembg": "loaded" if bg_remover.loaded_models() else "lazy",
            "upscaler": "loaded" if upscaler.is_ready() else ("loading" if upscaler.is_loading() else "lazy"),
            "upscaler_provider": upscaler.get_provider() if upscaler.is_ready() else "none",
        },
        "ffmpeg": "ok" if video_resizer.is_available() else "missing",
        "disk": {
            "total_gb": round(disk.total / 1024**3, 1),
            "free_gb": round(disk.free / 1024**3, 1),
            "tmp_mb": round(tmp_size / 1024**2, 2)
        }
    })


# ─── Pipeline: kirim hasil satu tab ke input tab lain ───
@app.route("/transfer", methods=["POST"])
@login_required
def transfer():
    """Salin sebuah output dari tab asal ke folder input tab tujuan (sesi sama)."""
    data = request.json or {}
    from_tab = data.get("from_tab")
    to_tab = data.get("to_tab")
    output_id = data.get("output_id")
    original = data.get("original") or output_id
    valid = {"bg", "upscale", "resize", "retouch"}
    if from_tab not in valid or to_tab not in valid or from_tab == to_tab:
        return jsonify({"error": "Tab tidak valid."}), 400
    if not output_id:
        return jsonify({"error": "output_id kosong."}), 400

    src = get_work_dir(from_tab) / "output" / output_id
    if not src.exists():
        return jsonify({"error": "File sumber tidak ditemukan."}), 404

    new_id = f"{uuid.uuid4().hex}{src.suffix.lower()}"
    shutil.copy2(src, get_work_dir(to_tab) / "input" / new_id)
    return jsonify({"id": new_id, "original": original})


@app.route("/transfer-batch", methods=["POST"])
@login_required
def transfer_batch():
    """Salin sekelompok output dari tab asal ke folder input tab tujuan sekaligus."""
    data = request.json or {}
    from_tab = data.get("from_tab")
    to_tab = data.get("to_tab")
    items = data.get("items") or []
    valid = {"bg", "upscale", "resize", "retouch"}
    if from_tab not in valid or to_tab not in valid or from_tab == to_tab:
        return jsonify({"error": "Tab tidak valid."}), 400
    if not items:
        return jsonify({"error": "Tidak ada item untuk dikirim."}), 400

    transferred = []
    src_dir = get_work_dir(from_tab) / "output"
    dst_dir = get_work_dir(to_tab) / "input"

    for item in items:
        output_id = item.get("output_id")
        original = item.get("original") or output_id
        if not output_id:
            continue
        src = src_dir / output_id
        if src.exists():
            new_id = f"{uuid.uuid4().hex}{src.suffix.lower()}"
            shutil.copy2(src, dst_dir / new_id)
            transferred.append({"id": new_id, "original": original})

    return jsonify({"status": "ok", "transferred": transferred})


# ══════════════════════════════════════════════
# TAB 1 — Remove BG
# ══════════════════════════════════════════════

@app.route("/bg/upload", methods=["POST"])
@login_required
def bg_upload():
    return handle_upload("bg")


@app.route("/bg/process/<file_id>", methods=["POST"])
@login_required
def bg_process(file_id):
    work = get_work_dir("bg")
    in_path = work / "input" / file_id
    if not in_path.exists():
        return jsonify({"error": "File tidak ditemukan."}), 404

    data = request.json if request.is_json else {}
    model_name = data.get("model", bg_remover.DEFAULT_MODEL)
    bg_color = data.get("bg_color")  # None = transparan, hex = warna solid
    out_name = in_path.stem + "_nobg.png"
    out_path = work / "output" / out_name

    result = bg_remover.process_image(in_path, out_path, model_name, bg_color)
    return jsonify(result), 200 if result["status"] == "ok" else 500


@app.route("/bg/preview-input/<file_id>")
@login_required
def bg_preview_input(file_id):
    p = get_work_dir("bg") / "input" / file_id
    if not p.exists():
        abort(404)
    return send_file(p)


@app.route("/bg/preview/<output_id>")
@login_required
def bg_preview(output_id):
    p = get_work_dir("bg") / "output" / output_id
    if not p.exists():
        abort(404)
    return send_file(p)  # auto-detect (PNG transparan / JPG solid)


@app.route("/bg/download/<output_id>")
@login_required
def bg_download(output_id):
    p = get_work_dir("bg") / "output" / output_id
    if not p.exists():
        abort(404)
    dl_name = request.args.get("name", output_id)
    return send_file(p, as_attachment=True, download_name=dl_name)


@app.route("/bg/download-all", methods=["POST"])
@login_required
def bg_download_all():
    work = get_work_dir("bg")
    ids = request.json.get("output_ids", [])
    if not ids:
        return jsonify({"error": "Tidak ada file."}), 400
        
    zip_id = uuid.uuid4().hex
    zip_path = work / f"{zip_id}.zip"
    
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for oid in ids:
            p = work / "output" / oid
            if p.exists():
                zf.write(p, arcname=oid)
                
    return send_file(zip_path, as_attachment=True, download_name="zaneva_nobg.zip",
                     mimetype="application/zip")


@app.route("/bg/clear", methods=["POST"])
@login_required
def bg_clear():
    work = get_work_dir("bg")
    count = 0
    for folder in ["input", "output"]:
        for p in (work / folder).iterdir():
            p.unlink(missing_ok=True)
            count += 1
    return jsonify({"status": "ok", "deleted_count": count})


@app.route("/bg/delete-output/<output_id>", methods=["POST"])
@login_required
def bg_delete_output(output_id):
    p = get_work_dir("bg") / "output" / output_id
    if p.exists():
        p.unlink()
        return jsonify({"status": "ok"})
    return jsonify({"error": "File tidak ditemukan."}), 404


# ══════════════════════════════════════════════
# TAB 2 — Upscale
# ══════════════════════════════════════════════

@app.route("/upscale/init", methods=["GET"])
@login_required
def upscale_init():
    """Lazy-load upscaler models. Called when Upscale tab is first opened."""
    if upscaler.is_ready():
        return jsonify({"status": "ready", "message": "Model sudah siap",
                        "provider": upscaler.get_provider()})
    if upscaler.is_loading():
        return jsonify({"status": "loading", "message": "Model sedang dimuat...",
                        "provider": "detecting"})
    # Trigger load in background thread so we don't block the response
    def _load():
        upscaler.init()
    threading.Thread(target=_load, daemon=True).start()
    return jsonify({"status": "loading", "message": "Memuat model Upscaler...",
                    "provider": "detecting"})


@app.route("/upscale/status", methods=["GET"])
@login_required
def upscale_status():
    if upscaler.is_ready():
        return jsonify({"status": "ready", "provider": upscaler.get_provider()})
    elif upscaler.is_loading():
        return jsonify({"status": "loading", "provider": "detecting"})
    else:
        return jsonify({"status": "idle", "provider": "none"})


@app.route("/upscale/upload", methods=["POST"])
@login_required
def upscale_upload():
    return handle_upload("upscale")


@app.route("/upscale/process/<file_id>", methods=["POST"])
@login_required
def upscale_process(file_id):
    if not upscaler.is_ready():
        return jsonify({"error": "Model sedang dimuat atau belum siap. Tunggu sebentar."}), 503

    work = get_work_dir("upscale")
    in_path = work / "input" / file_id
    if not in_path.exists():
        return jsonify({"error": "File tidak ditemukan."}), 404

    data = request.json or {}
    model_name = data.get("model", "RealESRGAN_x4plus")
    scale = int(data.get("scale", 4))
    if scale not in (2, 4):
        scale = 4

    out_dir = work / "output"
    result = upscaler.process_image(in_path, out_dir, model_name, scale)
    return jsonify(result), 200 if result["status"] == "ok" else 500


@app.route("/upscale/preview-input/<file_id>")
@login_required
def upscale_preview_input(file_id):
    p = get_work_dir("upscale") / "input" / file_id
    if not p.exists():
        abort(404)
    return send_file(p)


@app.route("/upscale/preview/<output_id>")
@login_required
def upscale_preview(output_id):
    p = get_work_dir("upscale") / "output" / output_id
    if not p.exists():
        abort(404)
    return send_file(p, mimetype="image/png")


@app.route("/upscale/download/<output_id>")
@login_required
def upscale_download(output_id):
    p = get_work_dir("upscale") / "output" / output_id
    if not p.exists():
        abort(404)
    dl_name = request.args.get("name", output_id)
    return send_file(p, as_attachment=True, download_name=dl_name)


@app.route("/upscale/download-all", methods=["POST"])
@login_required
def upscale_download_all():
    work = get_work_dir("upscale")
    ids = request.json.get("output_ids", [])
    if not ids:
        return jsonify({"error": "Tidak ada file."}), 400
        
    zip_id = uuid.uuid4().hex
    zip_path = work / f"{zip_id}.zip"
    
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for oid in ids:
            p = work / "output" / oid
            if p.exists():
                zf.write(p, arcname=oid)
                
    return send_file(zip_path, as_attachment=True, download_name="zaneva_upscaled.zip",
                     mimetype="application/zip")


@app.route("/upscale/clear", methods=["POST"])
@login_required
def upscale_clear():
    work = get_work_dir("upscale")
    count = 0
    for folder in ["input", "output"]:
        for p in (work / folder).iterdir():
            p.unlink(missing_ok=True)
            count += 1
    return jsonify({"status": "ok", "deleted_count": count})


@app.route("/upscale/delete-output/<output_id>", methods=["POST"])
@login_required
def upscale_delete_output(output_id):
    p = get_work_dir("upscale") / "output" / output_id
    if p.exists():
        p.unlink()
        return jsonify({"status": "ok"})
    return jsonify({"error": "File tidak ditemukan."}), 404


# ══════════════════════════════════════════════
# TAB 3 — Resize & Compress
# ══════════════════════════════════════════════

@app.route("/resize/upload", methods=["POST"])
@login_required
def resize_upload():
    return handle_upload("resize")


@app.route("/resize/process/<file_id>", methods=["POST"])
@login_required
def resize_process(file_id):
    work = get_work_dir("resize")
    in_path = work / "input" / file_id
    if not in_path.exists():
        return jsonify({"error": "File tidak ditemukan."}), 404

    data = request.json or {}
    presets = data.get("presets", ["shopee"])
    method = data.get("method", "crop")
    pad_color = data.get("pad_color", "#FFFFFF")
    quality = int(data.get("quality", 85))
    original_name = data.get("original_name", in_path.stem)

    out_dir = work / "output"
    result = resizer.process_image(
        in_path, out_dir, presets, method, pad_color, quality,
        original_name=original_name
    )
    return jsonify(result), 200 if result["status"] == "ok" else 500


@app.route("/resize/preview-input/<file_id>")
@login_required
def resize_preview_input(file_id):
    p = get_work_dir("resize") / "input" / file_id
    if not p.exists():
        abort(404)
    return send_file(p)


@app.route("/resize/preview/<output_id>")
@login_required
def resize_preview(output_id):
    p = get_work_dir("resize") / "output" / output_id
    if not p.exists():
        abort(404)
    return send_file(p)


@app.route("/resize/download/<output_id>")
@login_required
def resize_download(output_id):
    p = get_work_dir("resize") / "output" / output_id
    if not p.exists():
        abort(404)
    dl_name = request.args.get("name", output_id)
    return send_file(p, as_attachment=True, download_name=dl_name)


@app.route("/resize/download-all", methods=["POST"])
@login_required
def resize_download_all():
    work = get_work_dir("resize")
    data = request.json or {}
    output_ids = data.get("output_ids", [])
    stem_map = data.get("stem_map", {})
    if not output_ids:
        return jsonify({"error": "Tidak ada file."}), 400
        
    zip_id = uuid.uuid4().hex
    zip_path = work / f"{zip_id}.zip"
    
    resizer.build_zip(output_ids, work / "output", stem_map, zip_path)
    return send_file(zip_path, as_attachment=True, download_name="zaneva_resized.zip",
                     mimetype="application/zip")


@app.route("/resize/clear", methods=["POST"])
@login_required
def resize_clear():
    work = get_work_dir("resize")
    count = 0
    for folder in ["input", "output"]:
        for p in (work / folder).iterdir():
            p.unlink(missing_ok=True)
            count += 1
    return jsonify({"status": "ok", "deleted_count": count})


@app.route("/resize/delete-output/<output_id>", methods=["POST"])
@login_required
def resize_delete_output(output_id):
    p = get_work_dir("resize") / "output" / output_id
    if p.exists():
        p.unlink()
        return jsonify({"status": "ok"})
    return jsonify({"error": "File tidak ditemukan."}), 404


# ══════════════════════════════════════════════
# TAB 4 — Photo Retouch
# ══════════════════════════════════════════════

@app.route("/retouch/upload", methods=["POST"])
@login_required
def retouch_upload():
    return handle_upload("retouch")


@app.route("/retouch/process/<file_id>", methods=["POST"])
@login_required
def retouch_process(file_id):
    work = get_work_dir("retouch")
    in_path = work / "input" / file_id
    if not in_path.exists():
        return jsonify({"error": "File tidak ditemukan."}), 404

    data = request.json if request.is_json else {}
    do_upscale = data.get("upscale", True)
    scale = int(data.get("scale", 2))
    if scale not in (2, 4):
        scale = 2

    out_dir = work / "output"
    result = retoucher.process_image(in_path, out_dir, do_upscale, scale)
    return jsonify(result), 200 if result["status"] == "ok" else 500


@app.route("/retouch/preview-input/<file_id>")
@login_required
def retouch_preview_input(file_id):
    p = get_work_dir("retouch") / "input" / file_id
    if not p.exists():
        abort(404)
    return send_file(p)


@app.route("/retouch/preview/<output_id>")
@login_required
def retouch_preview(output_id):
    p = get_work_dir("retouch") / "output" / output_id
    if not p.exists():
        abort(404)
    return send_file(p, mimetype="image/png")


@app.route("/retouch/download/<output_id>")
@login_required
def retouch_download(output_id):
    p = get_work_dir("retouch") / "output" / output_id
    if not p.exists():
        abort(404)
    dl_name = request.args.get("name", output_id)
    return send_file(p, as_attachment=True, download_name=dl_name)


@app.route("/retouch/download-all", methods=["POST"])
@login_required
def retouch_download_all():
    work = get_work_dir("retouch")
    ids = request.json.get("output_ids", [])
    if not ids:
        return jsonify({"error": "Tidak ada file."}), 400
        
    zip_id = uuid.uuid4().hex
    zip_path = work / f"{zip_id}.zip"
    
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for oid in ids:
            p = work / "output" / oid
            if p.exists():
                zf.write(p, arcname=oid)
                
    return send_file(zip_path, as_attachment=True, download_name="zaneva_retouched.zip",
                     mimetype="application/zip")


@app.route("/retouch/clear", methods=["POST"])
@login_required
def retouch_clear():
    work = get_work_dir("retouch")
    count = 0
    for folder in ["input", "output"]:
        for p in (work / folder).iterdir():
            p.unlink(missing_ok=True)
            count += 1
    return jsonify({"status": "ok", "deleted_count": count})


@app.route("/retouch/delete-output/<output_id>", methods=["POST"])
@login_required
def retouch_delete_output(output_id):
    p = get_work_dir("retouch") / "output" / output_id
    if p.exists():
        p.unlink()
        return jsonify({"status": "ok"})
    return jsonify({"error": "File tidak ditemukan."}), 404


@app.route("/retouch/status", methods=["GET"])
@login_required
def retouch_status():
    """Check if upscaler is available for retouch pipeline."""
    has_upscaler = upscaler.is_ready()
    return jsonify({
        "upscaler_ready": has_upscaler,
        "upscaler_provider": upscaler.get_provider() if has_upscaler else "none",
    })


# ══════════════════════════════════════════════
# TAB 5 — Video Downscale
# ══════════════════════════════════════════════
# Beda dari tab gambar: encode video makan menit, bukan detik, jadi request
# tidak boleh menunggu. Pola yang dipakai sama seperti /upscale/init +
# /upscale/status — thread background + polling dari frontend.
#
# Catatan: registry job ini in-memory. Aman selama gunicorn jalan dengan
# `-w 1` (satu proses, lihat Dockerfile). Kalau worker dinaikkan, tiap worker
# akan punya dict sendiri dan polling bisa nyasar ke worker yang salah —
# saat itu registry harus pindah ke storage bersama (Redis/file).
VIDEO_JOBS = {}
_VIDEO_JOBS_LOCK = threading.Lock()
_VIDEO_JOB_TTL = 6 * 3600


def _video_job_set(job_id: str, **kv):
    with _VIDEO_JOBS_LOCK:
        job = VIDEO_JOBS.get(job_id)
        if job is not None:
            job.update(kv)


def _video_job_get(job_id: str):
    """Ambil job milik sesi ini saja, supaya job orang lain tidak bisa diintip."""
    with _VIDEO_JOBS_LOCK:
        job = VIDEO_JOBS.get(job_id)
        if job is None or job["sid"] != session.get("sid"):
            return None
        return dict(job)


def _video_prune_jobs():
    now = time.time()
    with _VIDEO_JOBS_LOCK:
        for jid in [k for k, v in VIDEO_JOBS.items()
                    if now - v["created"] > _VIDEO_JOB_TTL]:
            VIDEO_JOBS.pop(jid, None)


@app.route("/video/available", methods=["GET"])
@login_required
def video_available():
    return jsonify({"available": video_resizer.is_available(),
                    "reason": video_resizer.unavailable_reason()})


@app.route("/video/upload", methods=["POST"])
@login_required
def video_upload():
    if not video_resizer.is_available():
        return jsonify({"error": video_resizer.unavailable_reason()}), 503
    return handle_upload("video")


@app.route("/video/process/<file_id>", methods=["POST"])
@login_required
def video_process(file_id):
    if not video_resizer.is_available():
        return jsonify({"error": video_resizer.unavailable_reason()}), 503

    work = get_work_dir("video")
    in_path = work / "input" / file_id
    if not in_path.exists():
        return jsonify({"error": "File tidak ditemukan."}), 404

    data = request.json or {}
    target = str(data.get("target", video_resizer.DEFAULT_TARGET))
    quality = str(data.get("quality", video_resizer.DEFAULT_QUALITY))
    original_name = data.get("original_name") or file_id

    # Output hasil downscale hampir selalu lebih kecil dari input, tapi selama
    # encode keduanya ada di disk sekaligus. Pakai 1.2x input sebagai estimasi.
    ok, err = _check_disk(int(in_path.stat().st_size * 1.2))
    if not ok:
        return jsonify({"error": err}), 507

    _video_prune_jobs()
    job_id = uuid.uuid4().hex
    with _VIDEO_JOBS_LOCK:
        VIDEO_JOBS[job_id] = {
            "sid": session.get("sid"), "status": "queued", "percent": 0.0,
            "cancel": False, "result": None, "error": None, "created": time.time(),
        }

    out_dir = work / "output"

    def _cancelled():
        with _VIDEO_JOBS_LOCK:
            job = VIDEO_JOBS.get(job_id)
            return bool(job and job["cancel"])

    def _on_progress(pct):
        # Progress pertama baru muncul setelah job lolos antrean semaphore,
        # jadi ini sekaligus penanda transisi queued -> running.
        _video_job_set(job_id, status="running", percent=round(pct, 1))

    def _work():
        try:
            result = video_resizer.process_video(
                in_path, out_dir, target, quality, original_name,
                progress_cb=_on_progress, cancel_cb=_cancelled,
            )
        except Exception as e:
            _video_job_set(job_id, status="error", error=str(e))
            return
        if result["status"] == "ok":
            _video_job_set(job_id, status="done", percent=100.0, result=result)
        elif result["status"] == "skipped":
            _video_job_set(job_id, status="skipped", error=result.get("error"),
                           result=result)
        else:
            _video_job_set(job_id, status="error", error=result.get("error"))

    threading.Thread(target=_work, daemon=True).start()
    return jsonify({"job_id": job_id, "status": "queued"})


@app.route("/video/job/<job_id>", methods=["GET"])
@login_required
def video_job(job_id):
    job = _video_job_get(job_id)
    if job is None:
        return jsonify({"error": "Job tidak ditemukan."}), 404
    return jsonify({"status": job["status"], "percent": job["percent"],
                    "result": job["result"], "error": job["error"]})


@app.route("/video/cancel/<job_id>", methods=["POST"])
@login_required
def video_cancel(job_id):
    if _video_job_get(job_id) is None:
        return jsonify({"error": "Job tidak ditemukan."}), 404
    _video_job_set(job_id, cancel=True)
    return jsonify({"status": "ok"})


@app.route("/video/preview/<output_id>")
@login_required
def video_preview(output_id):
    p = get_work_dir("video") / "output" / output_id
    if not p.exists():
        abort(404)
    # conditional=True mengaktifkan HTTP Range, syarat agar <video> bisa
    # di-seek tanpa mengunduh seluruh file lebih dulu.
    return send_file(p, mimetype="video/mp4", conditional=True)


@app.route("/video/download/<output_id>")
@login_required
def video_download(output_id):
    p = get_work_dir("video") / "output" / output_id
    if not p.exists():
        abort(404)
    dl_name = request.args.get("name", output_id)
    return send_file(p, as_attachment=True, download_name=dl_name,
                     mimetype="video/mp4")


@app.route("/video/delete-output/<output_id>", methods=["POST"])
@login_required
def video_delete_output(output_id):
    p = get_work_dir("video") / "output" / output_id
    if p.exists():
        p.unlink()
        return jsonify({"status": "ok"})
    return jsonify({"error": "File tidak ditemukan."}), 404


@app.route("/video/clear", methods=["POST"])
@login_required
def video_clear():
    work = get_work_dir("video")
    count = 0
    for folder in ["input", "output"]:
        for p in (work / folder).iterdir():
            p.unlink(missing_ok=True)
            count += 1
    return jsonify({"status": "ok", "deleted_count": count})


# ─── Entry point ───
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
# Easypanel sets PORT=80, use it if available
    for p in [port, port + 1, port + 2]:
        try:
            print(f"[ImageKit] Starting on port {p}...")
            app.run(host="0.0.0.0", port=p, debug=False)
            break
        except OSError:
            print(f"[ImageKit] Port {p} busy, trying next...")
