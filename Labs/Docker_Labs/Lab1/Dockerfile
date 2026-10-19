# Slim Python base: much smaller image than the full python:3.x tag.
FROM python:3.12-slim

# Don't write .pyc files; flush stdout so `docker logs` shows output live.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OUTPUT_DIR=/app/output

WORKDIR /app

# Install dependencies first so this layer is cached across code changes.
COPY src/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ .

# Run as a non-root user; it owns the output directory the scripts write to.
RUN useradd --create-home appuser && mkdir -p "$OUTPUT_DIR" \
    && chown appuser:appuser "$OUTPUT_DIR"
USER appuser

# Trains by default; override to predict, e.g. `docker run ... python predict.py`.
CMD ["python", "train.py"]
