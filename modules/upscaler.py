"""Upscaler Real-ESRGAN via onnxruntime (tanpa torch/basicsr).

Bobot model = konversi langsung dari RealESRGAN_*.pth resmi (lihat
convert_to_onnx.py), jadi hasilnya identik dengan versi torch.
Tiling meniru RealESRGANer (tile 256 + pad 10) agar bebas seam dan
hemat RAM untuk gambar besar.

Provider priority: CUDA → DirectML → CPU (auto-detect).
"""
import threading
from pathlib import Path

import numpy as np
from PIL import Image

_lock = threading.Lock()
_gpu_sem = threading.Semaphore(1)   # hanya 1 inference GPU sekaligus → cegah OOM cublas
_sessions = {}       # model_name -> onnxruntime.InferenceSession (GPU/best)
_cpu_sessions = {}   # model_name -> onnxruntime.InferenceSession (CPU fallback)
_loading = False
_loaded = False
_active_provider = "CPUExecutionProvider"  # akan di-update saat init

MODEL_DIR = Path(__file__).parent.parent / "models"
ALLOWED_MODELS = ["RealESRGAN_x4plus", "RealESRGAN_x2plus"]
MODEL_SCALES = {"RealESRGAN_x4plus": 4, "RealESRGAN_x2plus": 2}
DEFAULT_MODEL = "RealESRGAN_x4plus"

# TILE=64: actual input ke model = 84×84px (64+2×10).
# Intermediate feature map ~64ch × 84×84 × float32 ≈ 1.7 MB/layer — aman di VRAM kecil.
# (TILE=128 → ~7 MB/layer → OOM pada GPU <4 GB)
TILE = 64
TILE_PAD = 10  # genap, agar dimensi tile tetap genap (syarat model x2)


def _get_best_providers() -> list:
    """Deteksi otomatis provider terbaik yang tersedia.
    Priority: CUDA (NVIDIA) → DirectML (Windows GPU universal) → CPU.

    CATATAN: TensorrtExecutionProvider sengaja DI-SKIP.
    TRT melakukan JIT-compile model saat pertama run — bisa timeout / crash
    untuk model kompleks seperti x4plus (23 RRDB blocks).
    CUDAExecutionProvider sudah GPU-accelerated tanpa overhead TRT.
    """
    import onnxruntime as ort
    available = ort.get_available_providers()
    # Buang TRT dan OpenVINO agar tidak dipilih secara otomatis
    safe = [p for p in available if p not in (
        "TensorrtExecutionProvider", "OpenVINOExecutionProvider"
    )]
    print(f"[Upscaler] Available: {available}")
    print(f"[Upscaler] Safe providers (no TRT): {safe}")

    if "CUDAExecutionProvider" in safe:
        print("[Upscaler] ✅ GPU: NVIDIA CUDA (tanpa TensorRT)")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    if "DmlExecutionProvider" in safe:
        print("[Upscaler] ✅ GPU: DirectML (Windows)")
        return ["DmlExecutionProvider", "CPUExecutionProvider"]

    print("[Upscaler] ⚪ Fallback: CPU")
    return ["CPUExecutionProvider"]


def _detect_active_provider(providers: list) -> str:
    """Kembalikan nama provider utama yang dipakai (bukan fallback)."""
    return providers[0]


def _load_model(model_name: str, providers: list):
    """Internal: load a single ONNX model dengan providers yang ditentukan.
    Selalu load CPU session terpisah sebagai fallback OOM."""
    import onnxruntime as ort

    onnx_path = MODEL_DIR / f"{model_name}.onnx"
    if not onnx_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {onnx_path}. "
            "Jalankan: python download_models.py"
        )
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    try:
        sess = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=providers)
    except Exception as e:
        # Jika GPU provider gagal (misal CUDA tidak cocok), fallback ke CPU
        print(f"[Upscaler] ⚠️ Provider {providers[0]} gagal ({e}), fallback CPU")
        sess = ort.InferenceSession(
            str(onnx_path), sess_options=opts,
            providers=["CPUExecutionProvider"]
        )

    _sessions[model_name] = sess
    print(f"[Upscaler] Model loaded: {model_name} | provider: {sess.get_providers()[0]}")

    # Load CPU fallback session (ringan, untuk OOM recovery)
    if sess.get_providers()[0] != "CPUExecutionProvider":
        cpu_opts = ort.SessionOptions()
        cpu_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        cpu_sess = ort.InferenceSession(
            str(onnx_path), sess_options=cpu_opts,
            providers=["CPUExecutionProvider"]
        )
        _cpu_sessions[model_name] = cpu_sess
        print(f"[Upscaler] CPU fallback loaded: {model_name}")
    else:
        _cpu_sessions[model_name] = sess  # sudah CPU, pakai yang sama


def init() -> dict:
    """Lazy-load all upscale models. Called only when Upscale tab is first accessed.
    Returns {status: 'ready'|'loading'|'error', message: str, provider: str}"""
    global _loading, _loaded, _active_provider

    with _lock:
        if _loaded:
            return {"status": "ready", "message": "Model sudah siap", "provider": _active_provider}
        if _loading:
            return {"status": "loading", "message": "Model sedang dimuat...", "provider": _active_provider}
        _loading = True

    try:
        providers = _get_best_providers()
        _active_provider = _detect_active_provider(providers)

        for m in ALLOWED_MODELS:
            _load_model(m, providers)

        # Verifikasi provider yang benar-benar dipakai setelah load
        if _sessions:
            first_sess = next(iter(_sessions.values()))
            actual = first_sess.get_providers()
            _active_provider = actual[0] if actual else providers[0]
            print(f"[Upscaler] ✅ Aktif menggunakan: {_active_provider}")

        with _lock:
            _loaded = True
            _loading = False
        return {"status": "ready", "message": "Model berhasil dimuat", "provider": _active_provider}
    except Exception as e:
        with _lock:
            _loading = False
        return {"status": "error", "message": str(e), "provider": "none"}


def is_ready() -> bool:
    return _loaded


def is_loading() -> bool:
    return _loading


def get_provider() -> str:
    """Kembalikan provider yang sedang aktif digunakan."""
    return _active_provider


def _enhance(sess, img: np.ndarray, model_scale: int) -> np.ndarray:
    """Tiled inference. img: float32 CHW RGB 0..1 -> CHW hasil model_scale x."""
    c, h, w = img.shape

    # Model x2 memakai pixel_unshuffle -> dimensi input wajib genap.
    pad_h = h % 2 if model_scale == 2 else 0
    pad_w = w % 2 if model_scale == 2 else 0
    if pad_h or pad_w:
        img = np.pad(img, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
    _, ph, pw = img.shape

    out = np.empty((c, ph * model_scale, pw * model_scale), dtype=np.float32)
    for y0 in range(0, ph, TILE):
        for x0 in range(0, pw, TILE):
            y1, x1 = min(y0 + TILE, ph), min(x0 + TILE, pw)
            py0, px0 = max(y0 - TILE_PAD, 0), max(x0 - TILE_PAD, 0)
            py1, px1 = min(y1 + TILE_PAD, ph), min(x1 + TILE_PAD, pw)

            inp = img[:, py0:py1, px0:px1][np.newaxis]
            pred = sess.run(None, {"input": inp})[0][0]

            oy = (y0 - py0) * model_scale
            ox = (x0 - px0) * model_scale
            out[:, y0 * model_scale:y1 * model_scale,
                x0 * model_scale:x1 * model_scale] = \
                pred[:, oy:oy + (y1 - y0) * model_scale,
                     ox:ox + (x1 - x0) * model_scale]

    return out[:, :h * model_scale, :w * model_scale]


def _clear_gpu_cache():
    """Bebaskan VRAM setelah setiap gambar selesai agar tidak OOM pada
    gambar berikutnya. Aman dipanggil meski tidak ada GPU."""
    try:
        import ctypes
        if "CUDAExecutionProvider" in _active_provider:
            # ONNX Runtime tidak expose cuMemFree langsung;
            # panggil via ort internal jika tersedia.
            pass
    except Exception:
        pass
    try:
        import gc
        gc.collect()
    except Exception:
        pass


def process_image(in_path: Path, out_dir: Path,
                  model_name: str = DEFAULT_MODEL, scale: int = 4) -> dict:
    """Upscale a single image. Returns dict with status and resolution info."""
    if not _loaded:
        return {"status": "error", "error": "Model belum siap. Buka tab Upscale dulu."}

    if model_name not in _sessions:
        return {"status": "error", "error": f"Model {model_name} tidak tersedia."}

    # Serialisasi GPU inference — cegah cublasCreate VRAM OOM saat batch
    acquired = _gpu_sem.acquire(timeout=300)  # tunggu max 5 menit
    if not acquired:
        return {"status": "error", "error": "GPU sedang sibuk terlalu lama, coba lagi."}

    try:
        img_pil = Image.open(in_path).convert("RGB")
        orig_w, orig_h = img_pil.size

        arr = np.asarray(img_pil, dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)  # HWC -> CHW

        model_scale = MODEL_SCALES[model_name]
        try:
            out = _enhance(_sessions[model_name], arr, model_scale)
        except Exception as gpu_err:
            err_str = str(gpu_err)
            # OOM / cublas / cudnn error → retry via CPU
            is_oom = any(k in err_str for k in (
                "AllocateRaw", "Failed to allocate", "CUBLAS failure",
                "CUDNN_STATUS", "cublasCreate", "cudnnCreate", "out of memory"
            ))
            if is_oom and model_name in _cpu_sessions:
                print(f"[Upscaler] ⚠️ GPU OOM, retry CPU: {gpu_err}")
                out = _enhance(_cpu_sessions[model_name], arr, model_scale)
            else:
                raise

        out = (np.clip(out, 0, 1).transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
        output_pil = Image.fromarray(out)

        target = (orig_w * scale, orig_h * scale)
        if output_pil.size != target:
            output_pil = output_pil.resize(target, Image.LANCZOS)
        new_w, new_h = output_pil.size

        final_name = f"{in_path.stem}_{scale}x.png"
        final_path = out_dir / final_name
        output_pil.save(final_path, format="PNG")
        try:
            in_path.unlink(missing_ok=True)
        except OSError:
            pass  # input masih dipakai (preview di Windows); dibersihkan auto-cleanup

        _clear_gpu_cache()  # bebaskan VRAM setelah setiap gambar
        _gpu_sem.release()

        return {
            "status": "ok",
            "output_id": final_name,
            "orig_res": f"{orig_w}×{orig_h}",
            "new_res": f"{new_w}×{new_h}",
            "provider": _active_provider,
        }
    except Exception as e:
        _gpu_sem.release()
        return {"status": "error", "error": str(e)}
