FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OPENBLAS_NUM_THREADS=1 \
    MALLOC_ARENA_MAX=2 \
    PATRIOTS_HOST=0.0.0.0 \
    PATRIOTS_PORT=8080

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg tesseract-ocr tesseract-ocr-heb \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY --chown=1000:1000 . /app
RUN mkdir -p /app/work && chown -R 1000:1000 /app/work

VOLUME /app/work
EXPOSE 8080
USER 1000:1000
CMD ["python", "app.py"]
