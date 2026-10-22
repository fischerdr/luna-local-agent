#!/usr/bin/env bash
# Entry point for the llama.cpp container.
# Loads server.env (mounted as a volume), discovers the model,
# and runs llama-server with all the arguments start.sh would use.
#
# Volume mounts:
#   /models        - model GGUF files (read-only)
#   /config        - server.env, server.local.env (read-only)
#   /data          - writable: slots, .api_key

set -euo pipefail

CONFIG="/config/server.env"
[ -f /config/server.local.env ] && CONFIG="${CONFIG} /config/server.local.env"

# --- load settings ----------------------------------------------------------
# shellcheck disable=SC1090
for f in $CONFIG; do
    [ -f "$f" ] && source "$f"
done

# Defaults (in case server.env is missing or empty)
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
CTX="${CTX:-16384}"
NGL="${NGL:-99}"
THREADS="${THREADS:-9}"
BATCH="${BATCH:-4096}"
UBATCH="${UBATCH:-2048}"
PARALLEL="${PARALLEL:-4}"
KV_UNIFIED="${KV_UNIFIED:-1}"
CTX_CHECKPOINTS="${CTX_CHECKPOINTS:-32}"
REASONING_FORMAT="${REASONING_FORMAT:-deepseek}"
REASONING_BUDGET="${REASONING_BUDGET:--1}"

# --- find model -------------------------------------------------------------
MMPROJ="${MMPROJ:-}"
MMPROJ_MATCH="${MMPROJ_MATCH:-}"

if [ -z "$MODEL" ] && [ -n "$MODEL_MATCH" ]; then
    # Search /models (and subdirs) for a matching GGUF file
    mapfile -t found < <(find /models -maxdepth 5 -iname "${MODEL_MATCH}.gguf" ! -iname "*mmproj*" 2>/dev/null | sort)
    if [ "${#found[@]}" -eq 0 ]; then
        echo "Error: no model matching '$MODEL_MATCH' found in /models" >&2
        exit 1
    fi
    if [ "${#found[@]}" -gt 1 ]; then
        echo "Error: multiple models match '$MODEL_MATCH' - set MODEL= in server.env:" >&2
        printf '  %s\n' "${found[@]}" >&2
        exit 1
    fi
    MODEL="${found[0]}"
fi

if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
    echo "Error: MODEL not set and no matching model found" >&2
    exit 1
fi

[ -n "$ALIAS" ] || ALIAS="$(basename "$MODEL" .gguf)"

# --- find mmproj (vision adapter) -------------------------------------------
if [ "$MMPROJ" != off ] && [ -z "$MMPROJ" ]; then
    mapfile -t mm < <(find "$(dirname "$MODEL")" -maxdepth 1 -iname "*mmproj*.gguf" 2>/dev/null | sort)
    if [ "${#mm[@]}" -eq 0 ] && [ -n "$MMPROJ_MATCH" ]; then
        mapfile -t mm < <(find /models -maxdepth 5 -iname "${MMPROJ_MATCH}.gguf" -iname "*mmproj*" 2>/dev/null | sort)
    fi
    MMPROJ="${mm[0]:-}"
fi

# --- api key ----------------------------------------------------------------
mkdir -p /data/slots
if [ ! -s /data/.api_key ]; then
    ( umask 077; head -c 32 /dev/urandom | base64 | tr -d '/+=\n' > /data/.api_key )
fi
chmod 600 /data/.api_key

# --- build args -------------------------------------------------------------
args=(
    -m "$MODEL" --alias "$ALIAS"
    --host "$HOST" --port "$PORT"
    -c "$CTX" -ngl "$NGL" -t "$THREADS"
    -b "$BATCH" -ub "$UBATCH"
    -np "$PARALLEL" --ctx-checkpoints "$CTX_CHECKPOINTS"
    -fa auto
    --jinja
    --reasoning-format "$REASONING_FORMAT" --reasoning-budget "$REASONING_BUDGET"
    --metrics --slots --slot-save-path /data/slots
    --api-key-file /data/.api_key --cors-origins localhost
)

[ "$KV_UNIFIED" = "1" ] && args+=(--kv-unified)
[ -n "$MMPROJ" ] && args+=(--mmproj "$MMPROJ")
[ -n "$EXTRA" ] && args+=($EXTRA)

echo "chat: $ALIAS"
echo "  $MODEL"
if [ -n "$MMPROJ" ]; then
    echo "  vision: $(basename "$MMPROJ")"
else
    echo "  vision: off"
fi
echo "  http://$HOST:$PORT  (Ctrl+C stops server)"
echo

# --- run --------------------------------------------------------------------
exec llama-server "${args[@]}" "$@"
