FROM python:3.11-slim

# System deps needed by scipy/metpy/herbie
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libeccodes-dev \
    libgeos-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Create persistent data directories
# Railway mounts a volume at /data — symlink ./data there
RUN mkdir -p /data/backtest /data/grib \
    && ln -sfn /data data_volume || true

# Ensure UTF-8 locale (fixes Windows encoding issues in logs)
ENV PYTHONIOENCODING=utf-8
ENV LANG=C.UTF-8
ENV PYTHONUNBUFFERED=1

# Herbie cache dir
ENV HERBIE_CACHE_DIR=/data/grib

CMD ["python", "-X", "utf8", "src/main.py"]
