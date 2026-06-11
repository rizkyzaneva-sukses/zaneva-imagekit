FROM python:3.12-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download BG models saat build (layer ini di-cache — tidak diulang
# setiap kali kode berubah, karena diletakkan SEBELUM `COPY . .`).
# isnet (~170MB, default) + birefnet (~930MB, opsi "Best" di dropdown).
RUN python -c "from rembg import new_session; new_session('isnet-general-use'); new_session('birefnet-general')"

# Kode + model upscaler ONNX (models/*.onnx ikut repo)
COPY . .

EXPOSE ${PORT:-5000}

CMD sh -c 'gunicorn -w 1 -b 0.0.0.0:${PORT:-5000} --timeout 300 app:app'
