"""Download bobot RealESRGAN .pth resmi ke folder models/.

HANYA dibutuhkan untuk konversi ulang ke ONNX (convert_to_onnx.py).
Runtime app memakai models/*.onnx yang sudah ikut repo — tidak perlu
menjalankan script ini untuk deploy/pakai biasa."""
import os
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODELS = {
    "RealESRGAN_x4plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "RealESRGAN_x2plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
}


def download(name, url):
    out = MODELS_DIR / name
    if out.exists():
        print(f"  [skip] {name} sudah ada ({out.stat().st_size // 1024 // 1024}MB)")
        return
    print(f"  [download] {name} ...")

    def progress(count, block_size, total_size):
        pct = count * block_size * 100 // total_size
        print(f"\r    {pct}%", end="", flush=True)

    urllib.request.urlretrieve(url, out, reporthook=progress)
    print(f"\r  [ok] {name} ({out.stat().st_size // 1024 // 1024}MB)")


if __name__ == "__main__":
    print("=== Download RealESRGAN Models ===")
    for name, url in MODELS.items():
        download(name, url)
    print("=== Selesai ===")
