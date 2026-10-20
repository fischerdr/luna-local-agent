# CPU fallback — runs everywhere, no GPU needed.
FROM python:3.12-slim

# espeak-ng is needed by kokoro for phonemization
RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak-ng && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch (lightweight, no CUDA bloat)
RUN pip install --no-cache-dir \
    "torch --index-url https://download.pytorch.org/whl/cpu" \
    ctranslate2 \
    kokoro>=0.9.4 \
    numpy>=1.24 \
    soundfile

WORKDIR /opt/kokoro-reader/server

# Default to auto-detect (will use CPU if no GPU available)
ENV KOKORO_HOST=0.0.0.0 \
    KOKORO_PORT=8899 \
    KOKORO_LANG=a \
    KOKORO_DEVICE= \
    KOKORO_VOICE=am_adam

EXPOSE 8899

COPY kokoro_server.py blends.json start_server.sh ./

CMD ["python", "kokoro_server.py"]
