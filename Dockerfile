# ── GPU-capable image: CUDA 12.2 runtime + Python 3.11 ──────────────────────
# onnxruntime-gpu (ditarik oleh rembg[gpu]) membutuhkan libcuda / libcublas
# yang sudah tersedia di base image ini.
# Tanpa GPU pun container tetap jalan — CUDA provider akan gracefully
# fallback ke CPUExecutionProvider secara otomatis.
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Install Python 3.11 + system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3-pip \
        libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1 && \
    python -m pip install --no-cache-dir --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download BG models saat build (layer ini di-cache — tidak diulang
# setiap kali kode berubah, karena diletakkan SEBELUM `COPY . .`).
# isnet (~170MB, default) + birefnet (~930MB, opsi "Best" di dropdown).
RUN python -c "from rembg import new_session; new_session('isnet-general-use'); new_session('birefnet-general')"

# Kode + model upscaler ONNX (models/*.onnx ikut repo)
COPY . .

EXPOSE ${PORT:-5000}

# gthread: satu worker dengan 2 thread — proses upscale panjang tidak blokir
# status-polling atau request lain. Timeout 600s untuk gambar besar x4plus.
CMD sh -c 'gunicorn -w 1 --worker-class gthread --threads 2 -b 0.0.0.0:${PORT:-5000} --timeout 600 --graceful-timeout 30 app:app'
