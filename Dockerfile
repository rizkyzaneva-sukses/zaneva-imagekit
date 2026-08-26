# ── GPU-capable image: CUDA 12.2 + cuDNN 9 + Python 3.11 ─────────────────
# onnxruntime-gpu membutuhkan:
#   1. libcublas, libcufft, libcurand  → sudah ada di CUDA runtime image
#   2. libcudnn9                       → install manual di bawah
# Tanpa GPU, container tetap jalan → otomatis fallback ke CPU.
# ──────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
# Batasi thread OpenMP/MKL/ONNX agar inference tidak monopoli semua CPU core.
# Tanpa ini: ONNX Runtime pakai 16 core → CPU 100% → server freezes.
# Sesuaikan ORT_NUM_THREADS dengan jumlah core yg dialokasikan di Easypanel.
# Nilai ini juga dibaca oleh _load_model() di modules/upscaler.py.
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
ENV ORT_NUM_THREADS=4
WORKDIR /app

# ── Step 1: Install cuDNN 9 + Python 3.11 + system deps ──
# cuDNN 9 wajib agar CUDAExecutionProvider di onnxruntime-gpu bisa aktif.
# Tanpa cuDNN → onnxruntime fallback ke CPU (libcudnn.so.9 not found).
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-venv python3-pip \
    libgl1 libglib2.0-0 \
    ffmpeg \
    libcudnn9-cuda-12 \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1 \
    && python -m pip install --no-cache-dir --upgrade pip

# ── Step 2: Install Python dependencies ──
# rembg[gpu] akan install onnxruntime-gpu (bukan onnxruntime biasa).
# onnxruntime-gpu sudah include CPUExecutionProvider sebagai fallback.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Step 3: Verifikasi GPU provider tersedia ──
# Build akan gagal jika onnxruntime-gpu tidak bisa detect CUDA libs.
# Ini mencegah deploy container yang diam-diam fallback ke CPU.
RUN python -c "\
import onnxruntime as ort; \
providers = ort.get_available_providers(); \
print('Available providers:', providers); \
assert 'CUDAExecutionProvider' in providers, \
'GAGAL: CUDAExecutionProvider tidak tersedia! Cek instalasi CUDA/cuDNN.'; \
print('OK: CUDAExecutionProvider tersedia')"

# ── Step 4: Pre-download BG remover models (cached layer) ──
# isnet (~170MB, default) + birefnet (~930MB, opsi "Best" di dropdown).
# Diletakkan SEBELUM `COPY . .` agar tidak re-download saat kode berubah.
RUN python -c "from rembg import new_session; new_session('isnet-general-use'); new_session('birefnet-general')"

# ── Step 5: Copy kode + model ONNX ──
COPY . .

EXPOSE ${PORT:-5000}

# gthread: 1 worker + 8 threads → banyak device bisa masuk antrian bersamaan.
# GPU inference tetap serial via _gpu_sem di upscaler.py — VRAM aman.
# Timeout 600s → x4plus gambar besar butuh waktu lebih lama.
CMD sh -c 'gunicorn -w 1 --worker-class gthread --threads 8 -b 0.0.0.0:${PORT:-5000} --timeout 600 --graceful-timeout 30 app:app'
