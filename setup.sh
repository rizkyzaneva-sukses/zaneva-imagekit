#!/bin/bash
set -e
echo "=== Zaneva ImageKit Setup ==="

# Torch TIDAK lagi dibutuhkan — upscaler sekarang jalan via onnxruntime
# (sudah terbawa rembg[cpu]) dengan model models/*.onnx yang ikut repo.

echo "[1/2] Installing requirements..."
pip install -r requirements.txt

echo "[2/2] Pre-loading rembg default model (isnet, ~170MB)..."
python -c "from rembg import new_session; new_session('isnet-general-use'); print('rembg OK')"

echo ""
echo "=== Setup selesai ==="
echo "Jalankan: python app.py"
echo "Default port: 5000"
echo "Jangan lupa copy .env.example ke .env dan isi APP_PASSWORD"
