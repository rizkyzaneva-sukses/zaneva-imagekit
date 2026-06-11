import sys
import types
import threading
from pathlib import Path
from PIL import Image

_lock = threading.Lock()
_models = {}  # key: model_name -> RealESRGANer instance
_loading = False
_loaded = False

MODEL_DIR = Path(__file__).parent.parent / "models"
ALLOWED_MODELS = ["RealESRGAN_x4plus", "RealESRGAN_x2plus"]
DEFAULT_MODEL = "RealESRGAN_x4plus"


def _patch_torchvision_compat():
    """basicsr meng-import torchvision.transforms.functional_tensor yang sudah
    dihapus di torchvision>=0.17. Daftarkan modul shim ke sys.modules agar
    import basicsr tetap berhasil tanpa mengubah file site-packages."""
    mod = "torchvision.transforms.functional_tensor"
    if mod in sys.modules:
        return
    import torchvision.transforms.functional as F
    shim = types.ModuleType(mod)
    shim.rgb_to_grayscale = F.rgb_to_grayscale
    sys.modules[mod] = shim


def _load_model(model_name: str):
    """Internal: load a single RealESRGAN model. Must be called under lock."""
    _patch_torchvision_compat()
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    if model_name == "RealESRGAN_x4plus":
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=23, num_grow_ch=32, scale=4)
        scale = 4
    else:  # RealESRGAN_x2plus
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=23, num_grow_ch=32, scale=2)
        scale = 2

    pth_path = MODEL_DIR / f"{model_name}.pth"
    if not pth_path.exists():
        raise FileNotFoundError(f"Model file not found: {pth_path}")

    upsampler = RealESRGANer(
        scale=scale,
        model_path=str(pth_path),
        model=model,
        tile=256,
        tile_pad=10,
        pre_pad=0,
        half=False,
        device="cpu",
    )
    _models[model_name] = upsampler
    print(f"[Upscaler] Model loaded: {model_name}")


def init() -> dict:
    """Lazy-load all upscale models. Called only when Upscale tab is first accessed.
    Returns {status: 'ready'|'loading'|'error', message: str}"""
    global _loading, _loaded

    with _lock:
        if _loaded:
            return {"status": "ready", "message": "Model sudah siap"}
        if _loading:
            return {"status": "loading", "message": "Model sedang dimuat..."}

        _loading = True

    try:
        for m in ALLOWED_MODELS:
            _load_model(m)
        with _lock:
            _loaded = True
            _loading = False
        return {"status": "ready", "message": "Model berhasil dimuat"}
    except Exception as e:
        with _lock:
            _loading = False
        return {"status": "error", "message": str(e)}


def is_ready() -> bool:
    return _loaded


def is_loading() -> bool:
    return _loading


def process_image(in_path: Path, out_dir: Path,
                  model_name: str = DEFAULT_MODEL, scale: int = 4) -> dict:
    """Upscale a single image. Returns dict with status and resolution info."""
    if not _loaded:
        return {"status": "error", "error": "Model belum siap. Buka tab Upscale dulu."}

    if model_name not in _models:
        return {"status": "error", "error": f"Model {model_name} tidak tersedia."}

    try:
        import cv2
        import numpy as np

        img_pil = Image.open(in_path)
        orig_w, orig_h = img_pil.size

        # Convert PIL -> numpy BGR for RealESRGAN
        img_np = cv2.cvtColor(np.array(img_pil.convert("RGB")), cv2.COLOR_RGB2BGR)

        upsampler = _models[model_name]
        output, _ = upsampler.enhance(img_np, outscale=scale)

        # Convert back BGR -> RGB -> PIL
        output_pil = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        new_w, new_h = output_pil.size

        suffix = f"_{scale}x"
        final_name = in_path.stem + suffix + ".png"
        final_path = out_dir / final_name

        output_pil.save(final_path, format="PNG")
        in_path.unlink(missing_ok=True)

        return {
            "status": "ok",
            "output_id": final_name,
            "orig_res": f"{orig_w}×{orig_h}",
            "new_res": f"{new_w}×{new_h}",
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}
