"""Smoke test untuk exe hasil build: login -> upload -> remove BG."""
import io
import sys
import time

import requests
from PIL import Image

BASE = "http://127.0.0.1:5050"

# Tunggu server siap (preload model bisa makan waktu)
for i in range(180):
    try:
        r = requests.get(f"{BASE}/status", timeout=2)
        print("Server siap:", r.json())
        break
    except Exception:
        time.sleep(2)
else:
    sys.exit("Server tidak kunjung siap")

s = requests.Session()
r = s.post(f"{BASE}/login", data={"password": "zaneva2024"}, allow_redirects=True)
assert "/login" not in r.url, f"Login gagal: {r.url}"
print("Login OK")

# Buat gambar test: kotak merah di tengah background putih
img = Image.new("RGB", (200, 200), "white")
for x in range(60, 140):
    for y in range(60, 140):
        img.putpixel((x, y), (220, 30, 30))
buf = io.BytesIO()
img.save(buf, "PNG")
buf.seek(0)

r = s.post(f"{BASE}/bg/upload", files={"photos": ("test.png", buf, "image/png")})
data = r.json()
assert data["accepted"], f"Upload gagal: {data}"
fid = data["accepted"][0]["id"]
print("Upload OK:", fid)

r = s.post(f"{BASE}/bg/process/{fid}", json={"model": "birefnet-general"}, timeout=300)
result = r.json()
print("Process result:", result)
assert result["status"] == "ok", f"Proses gagal: {result}"

r = s.get(f"{BASE}/bg/download/{result['output_id']}")
out = Image.open(io.BytesIO(r.content))
print(f"BG output: {out.format} {out.size} mode={out.mode}")
assert out.mode == "RGBA", "Output bukan PNG transparan"

# ─── Upscale (ONNX) ───
s.get(f"{BASE}/upscale/init")
for i in range(120):
    st = s.get(f"{BASE}/upscale/status").json()["status"]
    if st == "ready":
        break
    time.sleep(2)
else:
    sys.exit("Upscaler tidak kunjung ready")
print("Upscaler ready")

buf.seek(0)
r = s.post(f"{BASE}/upscale/upload", files={"photos": ("up.png", buf, "image/png")})
fid = r.json()["accepted"][0]["id"]
r = s.post(f"{BASE}/upscale/process/{fid}",
           json={"model": "RealESRGAN_x4plus", "scale": 4}, timeout=600)
result = r.json()
print("Upscale result:", result)
assert result["status"] == "ok", f"Upscale gagal: {result}"
r = s.get(f"{BASE}/upscale/download/{result['output_id']}")
out = Image.open(io.BytesIO(r.content))
assert out.size == (800, 800), f"Ukuran upscale salah: {out.size}"
print(f"Upscale output: {out.size}")

# ─── Resize ───
buf.seek(0)
r = s.post(f"{BASE}/resize/upload", files={"photos": ("rz.png", buf, "image/png")})
fid = r.json()["accepted"][0]["id"]
r = s.post(f"{BASE}/resize/process/{fid}",
           json={"presets": ["shopee"], "method": "crop", "quality": 85}, timeout=120)
result = r.json()
print("Resize result:", result)
assert result["status"] == "ok", f"Resize gagal: {result}"

print("SMOKE TEST PASSED")
