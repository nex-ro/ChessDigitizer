# ──────────────────────────────────────────────
# Chess Digitizer — Flask + YOLO (ultralytics)
# ──────────────────────────────────────────────
FROM python:3.11-slim

# Hindari .pyc & buffer stdout, biar log langsung muncul
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependensi sistem:
# - libgl1, libglib2.0-0, libsm6, libxext6, libxrender1 -> dibutuhkan OpenCV
# - fonts-dejavu-core -> font DejaVuSans.ttf yang dipakai render_board_png()
# - curl -> untuk healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        fonts-dejavu-core \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependensi python dulu (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code aplikasi
# Pastikan folder model/ (berisi best.pt) dan templates/ ikut ter-copy
COPY . .

# Buat user non-root demi keamanan
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

ENV PORT=5000 \
    CONF_THRESH=0.35 \
    IMG_SIZE=640

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:${PORT}/ || exit 1

# Gunakan gunicorn untuk produksi (lebih stabil dari app.run bawaan Flask)
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 120 app:app"]