FROM python:3.12-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download BG model during build (birefnet-general ~1GB)
RUN python -c "from rembg import new_session; new_session('birefnet-general')"

EXPOSE ${PORT:-5000}

CMD sh -c 'gunicorn -w 1 -b 0.0.0.0:${PORT:-5000} --timeout 300 app:app'
