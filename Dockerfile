FROM python:3.12-slim

WORKDIR /app

# System deps (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libffi-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY crypto/requirements.txt /app/crypto/requirements.txt
RUN pip install --no-cache-dir -r /app/crypto/requirements.txt

# Copy source code
COPY crypto/ /app/crypto/

# Data and state persist via volume mount
VOLUME ["/app/crypto_data"]

# Graceful shutdown: Docker sends SIGTERM, compose waits 30s (stop_grace_period)
STOPSIGNAL SIGTERM

ENTRYPOINT ["python", "-m", "crypto.crypto_main"]
