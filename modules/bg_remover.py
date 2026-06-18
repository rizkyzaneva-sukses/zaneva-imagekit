import os
import uuid
import threading
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from rembg import remove, new_session

ALLOWED_MODELS = ["isnet-general-use", "birefnet-general", "birefnet-portrait"]
# isnet jauh lebih ringan (~170MB vs ~930MB) dan cepat di CPU;
# birefnet tetap tersedia di dropdown untuk hasil maksimal.
DEFAULT_MODEL = "isnet-general-use"

# Cache sessions per model
_sessions = {}
_lock = threading.Lock()


def get_session(model_name: str):
    if model_name not in ALLOWED_MODELS:
        model_name = DEFAULT_MODEL
    with _lock:
        if model_name not in _sessions:
            print(f"[BG Remover] Loading model: {model_name} ...")
            _sessions[model_name] = new_session(model_name)
            print(f"[BG Remover] Model ready: {model_name}")
    return _sessions[model_name]


def loaded_models() -> list:
    return list(_sessions.keys())


def remove_shadow(result_img: Image.Image) -> Image.Image:
    """Remove residual shadow from background-removed image."""
    result_np = np.array(result_img)
    if result_np.shape[2] != 4:
        return result_img

    alpha = result_np[:, :, 3]
    rgb = result_np[:, :, :3]

    shadow_edge = (alpha > 0) & (alpha < 30)
    gray = np.mean(rgb, axis=2)
    dark_shadow = (alpha > 0) & (alpha < 100) & (gray < 100)
    faint_shadow = (alpha > 0) & (alpha < 15)

    shadow_mask = shadow_edge | dark_shadow | faint_shadow

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    shadow_mask = cv2.dilate(shadow_mask.astype(np.uint8), kernel, iterations=2).astype(bool)

    subject_mask = alpha > 200
    subject_dilated = cv2.dilate(subject_mask.astype(np.uint8), kernel, iterations=10).astype(bool)

    safe_to_remove = shadow_mask & ~subject_dilated
    result_np[safe_to_remove, 3] = 0

    return Image.fromarray(result_np)


def _parse_hex(hex_str: str) -> tuple:
    """Parse hex color '#RRGGBB' / '#RGB' -> (r, g, b). Default putih bila gagal."""
    hex_str = (hex_str or "").lstrip("#")
    if len(hex_str) == 3:
        hex_str = "".join(c * 2 for c in hex_str)
    try:
        return tuple(int(hex_str[i:i + 2], 16) for i in (0, 2, 4))
    except Exception:
        return (255, 255, 255)


def process_image(in_path: Path, out_path: Path, model_name: str = DEFAULT_MODEL,
                  bg_color: str = None) -> dict:
    """Remove background from a single image.
    bg_color None  -> hasil PNG transparan (default).
    bg_color hex   -> komposit ke warna solid, hasil JPG (mis. '#FFFFFF' untuk Shopee).
    Returns dict with status."""
    try:
        sess = get_session(model_name)
        img = Image.open(in_path).convert("RGBA")
        result = remove(img, session=sess)
        result = remove_shadow(result)

        if bg_color:
            rgb = _parse_hex(bg_color)
            canvas = Image.new("RGB", result.size, rgb)
            canvas.paste(result, mask=result.split()[3])  # alpha channel sebagai mask
            out_path = out_path.with_suffix(".jpg")
            canvas.save(out_path, format="JPEG", quality=95, optimize=True)
        else:
            result.save(out_path, format="PNG")

        try:
            in_path.unlink(missing_ok=True)
        except OSError:
            pass  # input masih dipakai (preview di Windows); dibersihkan auto-cleanup
        return {"status": "ok", "output_id": out_path.name}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def preload_default():
    """Pre-load default model at startup."""
    get_session(DEFAULT_MODEL)
