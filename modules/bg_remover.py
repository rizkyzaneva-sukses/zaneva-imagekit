import os
import uuid
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from rembg import remove, new_session

ALLOWED_MODELS = ["birefnet-general", "birefnet-portrait", "isnet-general-use"]
DEFAULT_MODEL = "birefnet-general"

# Cache sessions per model
_sessions = {}


def get_session(model_name: str):
    if model_name not in ALLOWED_MODELS:
        model_name = DEFAULT_MODEL
    if model_name not in _sessions:
        print(f"[BG Remover] Loading model: {model_name} ...")
        _sessions[model_name] = new_session(model_name)
        print(f"[BG Remover] Model ready: {model_name}")
    return _sessions[model_name]


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


def process_image(in_path: Path, out_path: Path, model_name: str = DEFAULT_MODEL) -> dict:
    """Remove background from a single image. Returns dict with status."""
    try:
        sess = get_session(model_name)
        img = Image.open(in_path).convert("RGBA")
        result = remove(img, session=sess)
        result = remove_shadow(result)
        result.save(out_path, format="PNG")
        in_path.unlink(missing_ok=True)
        return {"status": "ok", "output_id": out_path.name}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def preload_default():
    """Pre-load default model at startup."""
    get_session(DEFAULT_MODEL)
