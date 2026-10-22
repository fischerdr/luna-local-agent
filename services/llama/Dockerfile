# llama.cpp's own server image, for Luna's three llama roles: chat,
# embed and rerank (ROLE=, see entrypoint.sh). Nothing is added but the
# entrypoint. The build context is this folder alone (.containerignore),
# so no config.json, key or model from the repo can end up in it.
#
# Pinned by digest: tags move, and upstream only build-tests the GPU
# images. python -m services passes BACKEND and its BASE_DIGEST
# (services/images.py); the defaults are the cpu image, build b11804.
#
# No USER: rootless Podman maps container-root to you, and upstream's
# /app and the volumes expect root.
ARG BACKEND=server
ARG BASE_DIGEST=sha256:8c74db29f1a2bef3f0a331b52c4d6214601dfceed4bee0b94cc48b3b23ff257d
FROM ghcr.io/ggml-org/llama.cpp:${BACKEND}@${BASE_DIGEST}

# python -m services knows an image is one of ours, and current, by its
# labels (luna.image-version, luna.base-digest, luna.backend), never by
# the tag alone. They come from podman build --label: a LABEL built from
# an ARG here can be served stale from the build cache.
COPY --chmod=755 entrypoint.sh /luna/entrypoint.sh
ENTRYPOINT ["/luna/entrypoint.sh"]
