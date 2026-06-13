# ── SeeOurBook API ──
# FastAPI backend with heavy system deps (tesseract, ffmpeg, ghostscript)

FROM python:3.11-slim-bookworm

# Prevent Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-ara \
    tesseract-ocr-eng \
    ghostscript \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api/ ./api/

# Create uploads/text directories
RUN mkdir -p /app/text /app/uploads /app/documents

EXPOSE 8080

# Run with uvicorn — workers=1 because background jobs live in the same process
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
