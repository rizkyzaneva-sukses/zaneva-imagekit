"""Photo Retouch — OpenCV frequency separation + optional upscale.

Skin retouch uses ultra-subtle frequency separation to smooth skin while
preserving natural texture. Upscale is optional and auto-detects:
  - If upscaler module is loaded → use RealESRGAN ONNX
  - Otherwise → skip upscale, return retouched image at original resolution
"""
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

# Try to import upscaler for optional upscale step
_upscaler = None
try:
    from modules import upscaler as _upscaler
except ImportError:
    pass


def retouch_skin(input_path: Path, output_path: Path) -> dict:
    """Apply ultra-subtle frequency separation retouch to face area.

    Returns dict with status and face detection info.
    """
    try:
        img = cv2.imread(str(input_path))
        if img is None:
            return {"status": "error", "error": "Cannot read image"}

        img_f = img.astype(np.float32)
        h, w = img.shape[:2]

        # Detect faces
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        # Stricter detection for large images
        min_size = max(100, min(w, h) // 10)
        faces = face_cascade.detectMultiScale(
            gray, 1.1, 7, minSize=(min_size, min_size)
        )

        result = img_f.copy()
        faces_retouched = 0

        for (x, y, fw, fh) in faces:
            pad = int(fw * 0.15)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w, x + fw + pad)
            y2 = min(h, y + fh + pad)

            face = img_f[y1:y2, x1:x2].copy()
            fH, fW = face.shape[:2]

            # Ultra-subtle frequency separation
            low_freq = cv2.GaussianBlur(face, (0, 0), 12)
            high_freq = face - low_freq

            # Smooth low frequency gently
            low_smooth = cv2.GaussianBlur(low_freq, (0, 0), 20)
            low_u8 = np.clip(low_smooth, 0, 255).astype(np.uint8)
            low_u8 = cv2.bilateralFilter(low_u8, d=5, sigmaColor=30, sigmaSpace=30)

            # Keep 95% of texture (preserve pores)
            high_keep = high_freq * 0.95
            retouched = low_u8.astype(np.float32) + high_keep

            # Gradient mask: less smoothing on eyes, more on cheeks
            mask = np.ones((fH, fW), dtype=np.float32) * 0.5
            mask[:int(fH * 0.25), :] = 0.3   # eyes area
            mask[int(fH * 0.7):, :] = 0.4     # mouth area
            mask = cv2.GaussianBlur(mask, (61, 61), 20)
            mask_3ch = np.stack([mask] * 3, axis=-1)

            result[y1:y2, x1:x2] = retouched * mask_3ch + face * (1 - mask_3ch)
            faces_retouched += 1

        # No contrast boost — keep original tones
        result_u8 = np.clip(result, 0, 255).astype(np.uint8)
        cv2.imwrite(str(output_path), result_u8)

        return {
            "status": "ok",
            "faces_detected": len(faces),
            "faces_retouched": faces_retouched,
            "resolution": f"{w}×{h}",
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}


def process_image(in_path: Path, out_dir: Path,
                  do_upscale: bool = True, scale: int = 2) -> dict:
    """Full retouch pipeline: skin retouch + optional upscale.

    Args:
        in_path: Input image path
        out_dir: Output directory
        do_upscale: Whether to upscale after retouch
        scale: Upscale factor (2 or 4)

    Returns dict with status, output_id, and resolution info.
    """
    try:
        # Step 1: Skin retouch
        retouched_name = f"{in_path.stem}_retouched.png"
        retouched_path = out_dir / retouched_name
        retouch_result = retouch_skin(in_path, retouched_path)

        if retouch_result["status"] != "ok":
            return retouch_result

        # Step 2: Optional upscale
        if do_upscale and _upscaler and _upscaler.is_ready():
            # Use RealESRGAN via existing upscaler module
            model_name = "RealESRGAN_x4plus" if scale == 4 else "RealESRGAN_x2plus"
            upscale_result = _upscaler.process_image(
                retouched_path, out_dir, model_name, scale
            )
            if upscale_result["status"] == "ok":
                # Clean up intermediate retouched file
                try:
                    retouched_path.unlink(missing_ok=True)
                except OSError:
                    pass
                return {
                    "status": "ok",
                    "output_id": upscale_result["output_id"],
                    "orig_res": upscale_result["orig_res"],
                    "new_res": upscale_result["new_res"],
                    "faces_retouched": retouch_result["faces_retouched"],
                    "upscaled": True,
                    "provider": upscale_result.get("provider", "unknown"),
                }
            else:
                # Upscale failed, return retouched only
                pass

        # Return retouched without upscale
        img = Image.open(retouched_path)
        w, h = img.size
        return {
            "status": "ok",
            "output_id": retouched_name,
            "orig_res": f"{w}×{h}",
            "new_res": f"{w}×{h}",
            "faces_retouched": retouch_result["faces_retouched"],
            "upscaled": False,
            "provider": "none",
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}
