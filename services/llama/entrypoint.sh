#!/usr/bin/env bash
# One llama-server, in its container, as one of Luna's roles:
#
#   ROLE=chat     the model she talks with          (host port PORT, 8080)
#   ROLE=embed    embeddings, for recall by meaning (EMBED_PORT, 8081)
#   ROLE=rerank   the reranker                      (RERANK_PORT, 8082)
#
# The same arguments llama/start.sh builds on the host, from the same
# server.env names. python -m services reads server.env (and the rest)
# on the host and hands them in as an env-file; the defaults below are
# server.env's own, for a container started any other way.
#
# The model is settled on the host before this runs, never searched for
# here:
#   LUNA_HF=<user>/<repo>[:quant]   downloaded by -hf into /models (the
#                                   luna-models volume), once
#   LUNA_MODEL=/models-host/...     a file from your own models folder,
#              or /models/...       or one imported into the volume
#
# Secrets, never in the environment or argv: /run/secrets/api_key (the
# same key as llama/.api_key), and /run/secrets/hf_token if you have
# one. Anything after the role's own arguments goes to llama-server.
set -euo pipefail

BIN=/app/llama-server
KEY=/run/secrets/api_key
ROLE="${ROLE:-chat}"

[ -s "$KEY" ] || {
    echo "$ROLE: no API key at $KEY - python -m services setup makes one." >&2; exit 1; }

# The token is yours, for gated models. Compose mounts an empty file
# when there isn't one, which means the same as none.
if [ -s /run/secrets/hf_token ]; then
    HF_TOKEN="$(cat /run/secrets/hf_token)"
    export HF_TOKEN
fi

# -hf downloads land in the volume, so they survive a new container.
# LLAMA_CACHE is llama-server's own (tools/server/README.md; first in
# common/hf-cache.cpp's lookup); HF_HUB_CACHE, Hugging Face's standard
# name, is next in that lookup - there in case a later build drops the
# first.
export LLAMA_CACHE="${LLAMA_CACHE:-/models}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$LLAMA_CACHE}"

if [ -n "${LUNA_HF:-}" ]; then
    model=(-hf "$LUNA_HF")
    name="${LUNA_HF%%:*}"
    name="${name##*/}"
elif [ -n "${LUNA_MODEL:-}" ]; then
    [ -f "$LUNA_MODEL" ] || {
        echo "$ROLE: model file not found in the container: $LUNA_MODEL" >&2; exit 1; }
    model=(-m "$LUNA_MODEL")
    name="$(basename "$LUNA_MODEL" .gguf)"
else
    echo "$ROLE: no model - python -m services up sets LUNA_HF or LUNA_MODEL." >&2; exit 1
fi

# Inside, every role listens on 8080; the host port is Podman's -p. The
# image's own LLAMA_ARG_HOST/PORT would only earn a warning beside them.
unset LLAMA_ARG_HOST LLAMA_ARG_PORT

args=(
    "${model[@]}"
    --host 0.0.0.0 --port 8080
    -t "${THREADS:-9}"
    --api-key-file "$KEY" --cors-origins localhost
)

case "$ROLE" in
chat)
    args+=(
        --alias "${ALIAS:-$name}"
        -c "${CTX:-16384}" -ngl "${NGL:-99}"
        -b "${BATCH:-4096}" -ub "${UBATCH:-2048}"
        -np "${PARALLEL:-4}" --ctx-checkpoints "${CTX_CHECKPOINTS:-32}"
        -fa auto
        # Tool calling needs the model's own chat template.
        --jinja
        --reasoning-format "${REASONING_FORMAT:-deepseek}"
        --reasoning-budget "${REASONING_BUDGET:--1}"
        # Saved slots go to the luna-data volume.
        --metrics --slots --slot-save-path /data/slots
    )
    [ "${KV_UNIFIED:-1}" = 1 ] && args+=(--kv-unified)

    # Vision, as in start.sh: off, a file, or (with -hf) the repo's own
    # mmproj, which llama-server fetches by itself.
    if [ "${MMPROJ:-}" = off ]; then
        args+=(--no-mmproj)
    elif [ -n "${LUNA_MMPROJ:-}" ]; then
        [ -f "$LUNA_MMPROJ" ] || {
            echo "chat: mmproj file not found in the container: $LUNA_MMPROJ" >&2; exit 1; }
        args+=(--mmproj "$LUNA_MMPROJ")
    elif [ -n "${LUNA_MMPROJ_URL:-}" ]; then
        args+=(--mmproj-url "$LUNA_MMPROJ_URL")
    fi
    mkdir -p /data/slots

    # shellcheck disable=SC2206
    [ -n "${EXTRA:-}" ] && args+=($EXTRA)
    ;;
embed)
    # One fact is a sentence; the whole input must fit one batch, so the
    # batch is the context.
    ctx="${EMBED_CTX:-2048}"
    args+=(--alias "$name" --embeddings -c "$ctx" -b "$ctx" -ub "$ctx"
           -ngl "${EMBED_NGL:-0}")
    [ -n "${EMBED_POOLING-last}" ] && args+=(--pooling "${EMBED_POOLING-last}")
    ;;
rerank)
    # Each query-and-fact pair has to fit a batch.
    ctx="${RERANK_CTX:-2048}"
    args+=(--alias "$name" --reranking --pooling rank -c "$ctx" -b "$ctx" -ub "$ctx"
           -ngl "${RERANK_NGL:-99}")
    ;;
*)
    echo "ROLE must be chat, embed or rerank, not: $ROLE" >&2; exit 1
    ;;
esac

echo "$ROLE: $name"
exec "$BIN" "${args[@]}" "$@"
