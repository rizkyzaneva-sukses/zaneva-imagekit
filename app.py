import os
import uuid
import shutil
import threading
import time
import zipfile
from io import BytesIO
from pathlib import Path
from functools import wraps

from flask import (
    Flask, request, session, redirect, url_for,
    render_template, send_file, jsonify, abort
)
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "changeme-imagekit")

APP_PASSWORD = os.environ.get("APP_PASSWORD", "zaneva2025")
MAX_FILES = int(os.environ.get("MAX_FILES", 30))
MAX_FILE_MB = int(os.environ.get("MAX_FILE_SIZE_MB", 20))
TMP_BASE = Path("/tmp/imagekit")
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp"}
APP_VERSION = "1.0.0"

TMP_BASE.mkdir(exist_ok=True)

# ─── Import modules ───
from modules import bg_remover, upscaler, resizer

# Pre-load BG model at startup
print("[ImageKit] Pre-loading BG Remover default model...")
bg_remover.preload_default()
print("[ImageKit] Startup ready. Upscaler is lazy (not loaded yet).")


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


def _check_disk():
    stat = shutil.disk_usage("/")
    free_gb = stat.free / (1024 ** 3)
    if free_gb < 1.0:
        return False, f"Disk hampir penuh ({free_gb:.1f}GB tersisa)"
    return True, None


def handle_upload(tab: str):
    """Common upload handler for all tabs."""
    ok, err = _check_disk()
    if not ok:
        return jsonify({"error": err}), 507

    files = request.files.getlist("photos")
    if not files:
        return jsonify({"error": "Tidak ada file yang dikirim."}), 400

    work = get_work_dir(tab)
    accepted, rejected = [], []

    for f in files[:MAX_FILES]:
        ext = Path(f.filename).suffix.lower()
        if ext not in ALLOWED_EXT:
            rejected.append({"name": f.filename, "reason": "Format tidak didukung (JPG/PNG/WEBP)"})
            continue
        content = f.read()
        if len(content) > MAX_FILE_MB * 1024 * 1024:
            rejected.append({"name": f.filename, "reason": f"Ukuran melebihi {MAX_FILE_MB}MB"})
            continue
        safe_name = f"{uuid.uuid4().hex}{ext}"
        (work / "input" / safe_name).write_bytes(content)
        accepted.append({"id": safe_name, "original": f.filename})

    if len(files) > MAX_FILES:
        rejected.append({"name": "...", "reason": f"Hanya {MAX_FILES} file pertama yang diproses"})

    return jsonify({"accepted": accepted, "rejected": rejected})


# ─── Auth ───
@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        if request.form.get("password") == APP_PASSWORD:
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
                           version=APP_VERSION,
                           presets=resizer.PLATFORM_PRESETS)


# ─── Status ───
@app.route("/status")
def status():
    disk = shutil.disk_usage("/")
    tmp_size = sum(f.stat().st_size for f in TMP_BASE.rglob("*") if f.is_file())
    return jsonify({
        "version": APP_VERSION,
        "models": {
            "rembg": "loaded",
            "upscaler": "loaded" if upscaler.is_ready() else ("loading" if upscaler.is_loading() else "lazy")
        },
        "disk": {
            "total_gb": round(disk.total / 1024**3, 1),
            "free_gb": round(disk.free / 1024**3, 1),
            "tmp_mb": round(tmp_size / 1024**2, 2)
        }
    })


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

    model_name = request.json.get("model", "birefnet-general") if request.is_json else "birefnet-general"
    out_name = in_path.stem + "_nobg.png"
    out_path = work / "output" / out_name

    result = bg_remover.process_image(in_path, out_path, model_name)
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
    return send_file(p, mimetype="image/png")


@app.route("/bg/download/<output_id>")
@login_required
def bg_download(output_id):
    p = get_work_dir("bg") / "output" / output_id
    if not p.exists():
        abort(404)
    return send_file(p, as_attachment=True, download_name=output_id)


@app.route("/bg/download-all", methods=["POST"])
@login_required
def bg_download_all():
    work = get_work_dir("bg")
    ids = request.json.get("output_ids", [])
    if not ids:
        return jsonify({"error": "Tidak ada file."}), 400
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for oid in ids:
            p = work / "output" / oid
            if p.exists():
                zf.write(p, arcname=oid)
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name="zaneva_nobg.zip",
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
        return jsonify({"status": "ready", "message": "Model sudah siap"})
    if upscaler.is_loading():
        return jsonify({"status": "loading", "message": "Model sedang dimuat..."})
    # Trigger load in background thread so we don't block the response
    def _load():
        upscaler.init()
    threading.Thread(target=_load, daemon=True).start()
    return jsonify({"status": "loading", "message": "Memuat model Upscaler..."})


@app.route("/upscale/status", methods=["GET"])
@login_required
def upscale_status():
    if upscaler.is_ready():
        return jsonify({"status": "ready"})
    elif upscaler.is_loading():
        return jsonify({"status": "loading"})
    else:
        return jsonify({"status": "idle"})


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
    return send_file(p, as_attachment=True, download_name=output_id)


@app.route("/upscale/download-all", methods=["POST"])
@login_required
def upscale_download_all():
    work = get_work_dir("upscale")
    ids = request.json.get("output_ids", [])
    if not ids:
        return jsonify({"error": "Tidak ada file."}), 400
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for oid in ids:
            p = work / "output" / oid
            if p.exists():
                zf.write(p, arcname=oid)
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name="zaneva_upscaled.zip",
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
    return send_file(p, as_attachment=True, download_name=output_id)


@app.route("/resize/download-all", methods=["POST"])
@login_required
def resize_download_all():
    work = get_work_dir("resize")
    data = request.json or {}
    output_ids = data.get("output_ids", [])
    stem_map = data.get("stem_map", {})
    if not output_ids:
        return jsonify({"error": "Tidak ada file."}), 400
    buf = resizer.build_zip(output_ids, work / "output", stem_map)
    return send_file(buf, as_attachment=True, download_name="zaneva_resized.zip",
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
