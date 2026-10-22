# Kokoro, for Luna's voice: kokoro-reader's server, as the installer runs
# it on the host - POST /tts, GET /health, GET /voices on :8899 - in a
# container, so the host needs no Python for it.
#
# One base for cpu and cuda: torch's CUDA wheels bring their own CUDA
# runtime, and the driver comes from the host through CDI, so an
# nvidia/cuda base would only be a second copy. python -m services
# passes TORCH_INDEX (whl/cu126 or whl/cu130) for an NVIDIA machine; the
# default is the CPU build. Kokoro needs Python 3.10-3.12.
ARG PYTHON_DIGEST=sha256:cb33f7f771ecec2ef1225cf39b159591a040391bd73e4c01f00fdfc4c47611d2
FROM docker.io/library/python:3.12-slim@${PYTHON_DIGEST}

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG TORCH_VERSION=2.14.1
ARG KOKORO_REPO=https://github.com/fswolf/kokoro-reader.git
ARG KOKORO_REF=7c70135090f4cc228e0fc2d46900cd42d104a392

# The model (about 330 MB) downloads once, into the luna-tts-cache volume.
# KOKORO_HOST is 0.0.0.0 inside only: Podman publishes it on 127.0.0.1.
ENV PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/cache \
    KOKORO_HOST=0.0.0.0 \
    KOKORO_PORT=8899 \
    KOKORO_LANG=a

# espeak-ng: misaki's fallback for words it doesn't know.
RUN apt-get update \
 && apt-get install -y --no-install-recommends espeak-ng git ca-certificates \
 && git clone "$KOKORO_REPO" /app \
 && git -C /app checkout --detach "$KOKORO_REF" \
 && rm -rf /app/.git \
 && apt-get purge -y git \
 && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/*

# torch first, from its own index: PyPI's Linux wheel is the CUDA one.
RUN pip install --index-url "$TORCH_INDEX" "torch==${TORCH_VERSION}"

# Every pin quoted: unquoted, a >= is a shell redirect.
RUN pip install "kokoro==0.9.4" "misaki[en]==0.9.4" "numpy==2.5.4"

# misaki's English model: otherwise it pip-installs it on its first
# sentence, into a container that forgets it when it's made anew.
RUN python -m spacy download en_core_web_sm

WORKDIR /app/server
CMD ["python", "kokoro_server.py"]
