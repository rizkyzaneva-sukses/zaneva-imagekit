"""Test cepat modul upscaler ONNX (dev, tanpa server)."""
import tempfile
from pathlib import Path

from PIL import Image

from modules import upscaler

tmp = Path(tempfile.mkdtemp())

print("init:", upscaler.init())
assert upscaler.is_ready()

cases = [
    # (ukuran, model, scale) — odd dims menguji mod-pad x2, 300x260 menguji multi-tile
    ((120, 90), "RealESRGAN_x4plus", 4),
    ((101, 77), "RealESRGAN_x2plus", 2),
    ((300, 260), "RealESRGAN_x2plus", 2),
    ((80, 60), "RealESRGAN_x4plus", 2),  # outscale != model scale -> resize
]
for size, model, scale in cases:
    src = tmp / f"in_{size[0]}x{size[1]}_{scale}.png"
    img = Image.new("RGB", size)
    img.putdata([(x % 256, y % 256, (x + y) % 256) for y in range(size[1]) for x in range(size[0])])
    img.save(src)
    r = upscaler.process_image(src, tmp, model, scale)
    assert r["status"] == "ok", f"{size} {model} {scale}x: {r}"
    out = Image.open(tmp / r["output_id"])
    expect = (size[0] * scale, size[1] * scale)
    assert out.size == expect, f"{size}: dapat {out.size}, harusnya {expect}"
    print(f"OK {size} {model} {scale}x -> {out.size}")

print("UPSCALER DEV TEST PASSED")
