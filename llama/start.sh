#!/usr/bin/env bash
# Serve the model with llama.cpp's own server - and, when their models
# are there, the small embedding and reranker servers beside it. Luna
# finds all of them by herself.
#
#   llama/start.sh                 all of them (each side server only if its model is found)
#   llama/start.sh --verbose       anything extra goes to the chat server
#
# Chat on :8080 in this terminal; embeddings on :8081 and the reranker
# on :8082 in the background, logging to llama/embed.log and
# llama/rerank.log. Ctrl+C stops them all. Settings: server.env
# (server.mac.env on macOS), then server.local.env if present.
#
# Container mode (llama.cpp from pre-built image):
#   The container runs the main chat server; side servers run via
#   podman exec when no local llama-server binary exists.
#
set -uo pipefail

# macOS still ships bash 3.2, which has no mapfile. Hop to Homebrew's.
if [ "${BASH_VERSINFO[0]}" -lt 4 ]; then
    for b in /opt/homebrew/bin/bash /usr/local/bin/bash; do
        [ -x "$b" ] && exec "$b" "$0" "$@"
    done
    echo "This needs bash 4 or newer (macOS ships 3.2): brew install bash" >&2
    exit 1
fi

cd "$(dirname "$(realpath "$0" 2>/dev/null || echo "$0")")" || exit 1

# Settings: server.env (Linux/desktop) or server.mac.env on macOS, or
# LLAMA_ENV=<file> to pick one. Then server.local.env, if there is one,
# for this machine's own overrides (a MODEL path) - it isn't committed.
SERVER_ENV="${LLAMA_ENV:-server.env}"
if [ -z "${LLAMA_ENV:-}" ] && [ "$(uname)" = Darwin ] && [ -f server.mac.env ]; then
    SERVER_ENV=server.mac.env
fi
# shellcheck disable=SC1090
source "./$SERVER_ENV"
# shellcheck disable=SC1091
[ -f server.local.env ] && source ./server.local.env
echo "settings: llama/$SERVER_ENV$([ -f server.local.env ] && echo ' + server.local.env')"

# ---------------------------------------------------------------------------
# Container detection: use pre-built llama.cpp image if no local binary
# ---------------------------------------------------------------------------
LLAMA_CONTAINER=""
CONTAINER_MODE=false
LOCAL_BIN="./bin/llama-server"

if [ ! -x "$LOCAL_BIN" ]; then
    # No local binary — try the container
    if podman ps -a --filter "name=luna-llama" --format "{{.Status}}" 2>/dev/null | grep -qi "up"; then
        LLAMA_CONTAINER=1
        CONTAINER_MODE=true
        echo "llama: using container (luna-llama)"
    elif podman images -q luna-llama:cuda 2>/dev/null || \
         podman images -q luna-llama:vulkan 2>/dev/null; then
        LLAMA_CONTAINER=1
        CONTAINER_MODE=true
        echo "llama: container image exists, starting..."
        # Use Python to start the container (it reads server.env for model selection)
        cd "$ROOT" 2>/dev/null || true
        python3 -c "
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath('$0')))
import llama.llama_container as lc
if not lc.start_llama_container():
    sys.exit(1)
" 2>/dev/null
    fi
fi

# If container mode is active but no container is running, try to start one
if [ "$CONTAINER_MODE" = true ] && [ -z "$LLAMA_CONTAINER" ]; then
    podman ps -a --filter "name=luna-llama" --format "{{.Status}}" 2>/dev/null | grep -qi "up" && LLAMA_CONTAINER=1
fi

# --- find_model PATTERN -> the one GGUF matching it in llama/models or LM
# Studio's model folders (normal install, old cache, Flatpak). Prints
# why on stderr and fails if there's none, or more than one to choose
# between. mmproj files are vision adapters, never models.
find_model() {
    local pattern="$1" found=()

    # Also search inside container's model volume
    local extra_search=""
    if [ "$CONTAINER_MODE" = true ]; then
        extra_search="/models"
    fi

    mapfile -t found < <(
        find ./models "$HOME/.lmstudio/models" "$HOME/.cache/lm-studio/models" \
             "$HOME/.var/app/ai.lmstudio.lm-studio/.lmstudio/models" \
             $extra_search \
             -maxdepth 5 -iname "${pattern}.gguf" ! -iname "*mmproj*" \
             2>/dev/null | sort
    )

    if [ "${#found[@]}" -eq 0 ]; then
        echo "No GGUF matching $pattern in llama/models or LM Studio's model folders." >&2
        return 1
    fi

    if [ "${#found[@]}" -gt 1 ]; then
        echo "More than one file matches $pattern - set the path in llama/server.env:" >&2
        printf '  %s\n' "${found[@]}" >&2
        return 1
    fi

    echo "${found[0]}"
}

serving() { curl -fsS --max-time 1 "http://$HOST:$1/health" >/dev/null 2>&1; }

# --- the chat model -------------------------------------------------------
# In container mode, the container finds the model itself. Here we just
# verify one is available and let the container handle it.
if [ "$CONTAINER_MODE" = true ]; then
    # Verify a model exists (container will find it)
    mapfile -t models < <(find ./models ./ -maxdepth 1 -iname "*.gguf" ! -iname "*mmproj*" 2>/dev/null | sort)
    if [ "${#models[@]}" -eq 0 ] && [ -n "$MODEL_MATCH" ]; then
        model_path="$(find_model "$MODEL_MATCH" 2>/dev/null)" || {
            echo "No model found - set MODEL= in llama/server.env to the GGUF file path." >&2
            exit 1
        }
    elif [ "${#models[@]}" -eq 0 ]; then
        echo "No model found - drop a GGUF file in llama/models/" >&2
        exit 1
    fi
    ALIAS="${ALIAS:-$(basename "${models[0]}" .gguf)}"
    echo "chat: container mode - $ALIAS (from llama/models)"
else
    [ -n "$MODEL" ] || MODEL="$(find_model "$MODEL_MATCH")" || {
        echo "Set MODEL= in llama/server.env to the file's full path." >&2; exit 1; }
    [ -f "$MODEL" ] || { echo "Model file not found: $MODEL" >&2; exit 1; }
    [ -n "$ALIAS" ] || ALIAS="$(basename "$MODEL" .gguf)"
fi

# The vision adapter. LM Studio downloads it beside the model as
# mmproj-*.gguf, so the model's own folder is the first place to look.
MMPROJ="${MMPROJ:-}"
MMPROJ_MATCH="${MMPROJ_MATCH:-}"
if [ "$MMPROJ" = off ]; then
    MMPROJ=""
elif [ -z "$MMPROJ" ] && [ "$CONTAINER_MODE" != true ]; then
    mapfile -t mm < <(find "$(dirname "$MODEL")" -maxdepth 1 -iname "*mmproj*.gguf" 2>/dev/null | sort)
    if [ "${#mm[@]}" -eq 0 ] && [ -n "$MMPROJ_MATCH" ]; then
        mapfile -t mm < <(find ./models "$HOME/.lmstudio/models" "$HOME/.cache/lm-studio/models" \
                               "$HOME/.var/app/ai.lmstudio.lm-studio/.lmstudio/models" \
                               -maxdepth 5 -iname "${MMPROJ_MATCH}.gguf" -iname "*mmproj*" 2>/dev/null | sort)
    fi
    if [ "${#mm[@]}" -gt 1 ]; then
        echo "More than one mmproj found - set MMPROJ= in llama/server.env:" >&2
        printf '  %s\n' "${mm[@]}" >&2
        exit 1
    fi
    MMPROJ="${mm[0]:-}"
elif [ -n "$MMPROJ" ] && [ "$CONTAINER_MODE" != true ] && [ ! -f "$MMPROJ" ]; then
    echo "MMPROJ in server.env doesn't exist: $MMPROJ" >&2; exit 1
fi

serving "$PORT" && { echo "Something is already serving on $HOST:$PORT." >&2; exit 1; }

# One copy of the model fits in VRAM (or a Mac's shared memory), not two.
if curl -fsS --max-time 2 http://localhost:1234/api/v0/models 2>/dev/null |
   grep -q '"state": *"loaded"'; then
    echo "LM Studio has a model loaded - eject it first, or both won't fit in VRAM." >&2
    echo >&2
fi

# The API key. Made once, random, readable only by you; Luna reads the
# same file. Without one, any web page open in your browser can talk to
# a server on localhost - and this one can write files (saved slots)
# and keep the GPU busy. CORS is limited to localhost on top of it.
if [ ! -s .api_key ] && [ "$CONTAINER_MODE" != true ]; then
    ( umask 077; head -c 32 /dev/urandom | base64 | tr -d '/+=\n' > .api_key )
    echo "Made llama/.api_key - Luna picks it up on her next start." >&2
fi

# --- the side servers: embeddings, and the reranker -----------------------
# Each is a small model on its own port, in the background. Its switch
# in server.env: auto starts it when the model is found and says so
# when it isn't; off never starts it; on refuses to start without it.
side_pids=()

# A background job in a script ignores Ctrl+C, so they have to be
# stopped by hand - on Ctrl+C, on kill, and when the chat server exits
# by itself. Bash won't reliably run an EXIT trap after an interrupt,
# hence the separate INT/TERM one.
stop_sides() {
    [ "${#side_pids[@]}" -gt 0 ] && kill "${side_pids[@]}" 2>/dev/null
    side_pids=()
}
trap stop_sides EXIT
trap 'stop_sides; exit 130' INT TERM

# side_server LABEL SWITCH MODEL MATCH PORT LOG FALLBACK -- server args...
# MODEL may be empty (found by MATCH). FALLBACK is what Luna does
# without it, for the "off" line.
side_server() {
    local label="$1" switch="${2:-auto}" model="$3" match="$4" port="$5" log="$6" fallback="$7" why=""
    shift 8

    [ "$switch" = off ] && return 0

    if [ -z "$model" ]; then
        model="$(find_model "$match" 2>/tmp/.side_why.$$)" || model=""
        why="$(cat /tmp/.side_why.$$ 2>/dev/null)"; rm -f /tmp/.side_why.$$
    elif [ ! -f "$model" ]; then
        why="the model path set in server.env doesn't exist: $model"
        model=""
    fi

    if [ -z "$model" ]; then
        [ "$switch" = on ] && { echo "$label: $why" >&2; exit 1; }
        echo "$label: off - $why"
        echo "  ($fallback)"
        return 0
    fi

    if serving "$port"; then
        echo "$label: something is already on :$port - leaving it be."
        return 0
    fi

    # In container mode, side servers are managed on-demand by embedmem
    # (llama_side.py) — they start on first retrieval and stop after
    # IDLE_TIMEOUT seconds of inactivity. No podman exec here.
    echo "$label: on-demand (llama_side.py starts on first use)"
    return 0

    # Local binary mode
    "$LOCAL_BIN" -m "$model" --alias "$(basename "$model" .gguf)" \
        --host "$HOST" --port "$port" -ngl "$NGL" -t "$THREADS" \
        --api-key-file .api_key --cors-origins localhost \
        "$@" > "$log" 2>&1 &
    side_pids+=($!)
    echo "$label: $(basename "$model" .gguf) on http://$HOST:$port (log: llama/$log)"
}

# One fact is a sentence; the whole input must fit one batch, so the
# batch is the context.
eargs=(--embeddings -c "$EMBED_CTX" -b "$EMBED_CTX" -ub "$EMBED_CTX")
[ -n "$EMBED_POOLING" ] && eargs+=(--pooling "$EMBED_POOLING")
side_server embeddings "${EMBED:-auto}" "$EMBED_MODEL" "$EMBED_MATCH" "$EMBED_PORT" embed.log \
    "fact recall falls back to matching words; EMBED=off in server.env hides this" -- "${eargs[@]}"

# The reranker scores what you said and one fact together, so each pair
# has to fit a batch. RERANK=off hides the line when it isn't wanted.
RERANK_CTX="${RERANK_CTX:-2048}"
side_server reranker "${RERANK:-auto}" "${RERANK_MODEL:-}" "${RERANK_MATCH:-*qwen3-reranker*}" \
    "${RERANK_PORT:-8082}" rerank.log \
    "recall uses the embedding order as it is; RERANK=off in server.env hides this" -- \
    --reranking --pooling rank -c "$RERANK_CTX" -b "$RERANK_CTX" -ub "$RERANK_CTX"

# --- the chat server, in the foreground -----------------------------------
if [ "$CONTAINER_MODE" = true ] && [ -n "$LLAMA_CONTAINER" ]; then
    echo "  http://$HOST:$PORT  (Ctrl+C stops them all)"
    echo
    echo "  (server output from container - Ctrl+C stops everything)"
    echo

    # Stop side servers on exit (container mode: on-demand servers stop
    # via idle timeout; this is a no-op since side_pids is empty).
    stop_sides
    trap 'podman stop luna-llama 2>/dev/null; exit 130' INT TERM

    # The container's entrypoint runs llama-server with the model.
    # Just stream its logs to this terminal.
    exec podman logs -f luna-llama 2>&1
else
    args=(
        -m "$MODEL" --alias "$ALIAS"
        --host "$HOST" --port "$PORT"
        -c "$CTX" -ngl "$NGL" -t "$THREADS"
        -b "$BATCH" -ub "$UBATCH"
        -np "$PARALLEL" --ctx-checkpoints "$CTX_CHECKPOINTS"
        -fa auto
        # Tool calling needs the model's own chat template, not a built-in one.
        --jinja
        --reasoning-format "$REASONING_FORMAT" --reasoning-budget "$REASONING_BUDGET"
        # The research side: /metrics, /slots, and somewhere to save a slot's
        # KV cache so a long conversation can be resumed without re-reading it.
        --metrics --slots --slot-save-path ./slots
        --api-key-file .api_key --cors-origins localhost
    )

    [ "$KV_UNIFIED" = 1 ] && args+=(--kv-unified)
    [ -n "$MMPROJ" ] && args+=(--mmproj "$MMPROJ")

    # shellcheck disable=SC2206
    [ -n "$EXTRA" ] && args+=($EXTRA)

    # From the environment rather than server.env: introspection/blind.py
    # passes a control vector here for one session without editing a file.
    # shellcheck disable=SC2206
    [ -n "${EXTRA_ARGS:-}" ] && args+=($EXTRA_ARGS)

    echo "chat: $ALIAS"
    echo "  $MODEL"
    if [ -n "$MMPROJ" ]; then
        echo "  vision: $(basename "$MMPROJ")"
    else
        echo "  vision: off - no mmproj next to the model (see MMPROJ in server.env)"
    fi
    echo "  http://$HOST:$PORT  (Ctrl+C stops them all)"
    echo

    # Not exec: this shell has to outlive the server, to stop the side
    # servers when it goes.
    #
    # QUIET_LOG sends the server's own output to a file instead of this
    # terminal. A blind session needs it: llama-server announces the control
    # vector it loaded, which would give the game away.
    if [ -n "${QUIET_LOG:-}" ]; then
        echo "  (server output is going to ${QUIET_LOG} - don't read it until you reveal)"
        "$LOCAL_BIN" "${args[@]}" "$@" > "$QUIET_LOG" 2>&1
    else
        "$LOCAL_BIN" "${args[@]}" "$@"
    fi
fi
