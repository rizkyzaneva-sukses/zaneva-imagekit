#!/bin/bash
set -e
echo "=== Zaneva ImageKit Setup ==="

# Install PyTorch CPU
echo "[1/4] Installing PyTorch (CPU)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install requirements + Upscale deps (Real-ESRGAN).
# Catatan: basicsr vs torchvision>=0.17 sudah ditangani oleh shim di
# modules/upscaler.py (_patch_torchvision_compat), jadi tidak perlu pin versi.
echo "[2/4] Installing requirements + Real-ESRGAN..."
pip install -r requirements.txt
pip install realesrgan basicsr

# Download models
echo "[3/4] Downloading RealESRGAN models..."
python download_models.py

# Verify rembg
echo "[4/4] Pre-loading rembg model..."
python -c "from rembg import new_session; new_session('birefnet-general'); print('rembg OK')"

echo ""
echo "=== Setup selesai ==="
echo "Jalankan: python app.py"
echo "Default port: 5000"
echo "Jangan lupa copy .env.example ke .env dan isi APP_PASSWORD"
