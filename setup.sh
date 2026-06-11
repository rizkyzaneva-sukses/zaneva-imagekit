#!/bin/bash
set -e
echo "=== Zaneva ImageKit Setup ==="

# Install PyTorch CPU
echo "[1/4] Installing PyTorch (CPU)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install requirements
echo "[2/4] Installing requirements..."
pip install -r requirements.txt

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
