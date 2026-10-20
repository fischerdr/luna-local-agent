# CPU fallback — runs everywhere, no GPU needed.
FROM python:3.12-slim

# git + espeak-ng for kokoro phonemization
# Remove EXTERNALLY-MANAGED so pip works without --break-system-packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git espeak-ng && rm -rf /var/lib/apt/lists/* \
    && rm -f /usr/lib/python3.12/EXTERNALLY-MANAGED 2>/dev/null; true

# Clone kokoro-reader server files
RUN git clone --depth 1 https://github.com/fswolf/kokoro-reader.git /opt/kokoro-reader

# CPU-only PyTorch (lightweight, no CUDA bloat)
# --break-system-packages: standard in Docker since the container IS the environment
RUN pip install --no-cache-dir --break-system-packages \
    "torch[cpu]" \
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

CMD ["python3", "kokoro_server.py"]
