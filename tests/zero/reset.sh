#!/usr/bin/env bash
# Back to zero, for development: this checkout and this machine as they
# were just after git clone, so an install of the backend services can be
# tested from scratch and real-world problems show up. Nothing is kept or
# backed up - it's a test bench. Run it, then install as a new user would.
#
#   tests/zero/reset.sh                   show what goes, ask, reset
#   tests/zero/reset.sh --install         reset, then install with a new user's
#                                         defaults (--yes), start the containers,
#                                         check everything (tests/zero/check.sh)
#   tests/zero/reset.sh --install --interactive   the same, answering the installer yourself
#   tests/zero/reset.sh --yes ...         don't ask first
#
# What goes:
#   in the checkout - every gitignored file (.venv, install.json, config,
#     llama/.api_key, *.local.env, llama/models, deploy's .env and keys,
#     agent/'s databases, __pycache__), and the tracked files a run changes
#     (config.json, agent/) back as committed. Source files are never touched,
#     committed or not.
#   her data and logs - ~/.cache/ai-voice
#   the launchers - ~/.local/bin/luna and the app entry, when they're this checkout's
#   the host voice's venv - <kokoro-reader>/server/.venv, if the installer made one
#   Podman - every luna-* container, volume (the models with them) and
#     network, every luna_* secret, our images (luna-*), upstream's llama.cpp
#     images and python:3.12-slim, the env-files
#
# What stays: system packages, your Hugging Face cache and token
# (~/.cache/huggingface), LM Studio's models, the kokoro-reader clone itself,
# and anything that isn't Luna's (other containers and images).
set -euo pipefail
cd "$(dirname "$(realpath "$0")")/../.."
ROOT=$(pwd)

YES=0 INSTALL=0 INTERACTIVE=0
for a in "$@"; do
    case "$a" in
    --yes) YES=1 ;;
    --install) INSTALL=1 ;;
    --interactive) INTERACTIVE=1 ;;
    -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $a (see --help)" >&2; exit 2 ;;
    esac
done

have_podman=0
command -v podman >/dev/null 2>&1 && have_podman=1

# --- what there is -------------------------------------------------------------
# install.json says where the installer put things outside the checkout;
# read it before it goes.
state() {
    python3 -c 'import json, sys; print(json.load(open("install.json")).get(sys.argv[1], ""))' "$1" 2>/dev/null || true
}
venv=$(state venv)
kokoro=$(state kokoro_dir)

mapfile -t ignored < <(git clean -Xdn | sed 's/^Would remove //')
mapfile -t data < <(git status --porcelain -- config.json agent | awk '{print $2}')
outside=()
[ -d "${XDG_CACHE_HOME:-$HOME/.cache}/ai-voice" ] && outside+=("${XDG_CACHE_HOME:-$HOME/.cache}/ai-voice")
[ "$(readlink -f "$HOME/.local/bin/luna" 2>/dev/null)" = "$ROOT/start.sh" ] && outside+=("$HOME/.local/bin/luna")
desktop="$HOME/.local/share/applications/luna.desktop"
[ -f "$desktop" ] && grep -q "$ROOT" "$desktop" && outside+=("$desktop")
case "$venv" in "$ROOT"/*|"") ;; *) [ -d "$venv" ] && outside+=("$venv") ;; esac
[ -n "$kokoro" ] && [ -d "$kokoro/server/.venv" ] && outside+=("$kokoro/server/.venv")
[ -d "$HOME/.cache/luna-zero" ] && outside+=("$HOME/.cache/luna-zero")

containers=() volumes=() secrets=() networks=() images=()
if [ "$have_podman" = 1 ]; then
    mapfile -t containers < <(podman ps -a --format '{{.Names}}' | grep '^luna-' || true)
    mapfile -t volumes < <(podman volume ls --format '{{.Name}}' | grep '^luna-' || true)
    mapfile -t secrets < <(podman secret ls --format '{{.Name}}' | grep '^luna_' || true)
    mapfile -t networks < <(podman network ls --format '{{.Name}}' | grep '^luna' || true)
    mapfile -t images < <( {
        podman images -a -q --filter label=luna.kind
        podman images -a --format '{{.Repository}} {{.Tag}} {{.ID}}' | awk '
            $1 ~ /^localhost\/luna-/ || $1 == "ghcr.io/ggml-org/llama.cpp" ||
            ($1 == "docker.io/library/python" && $2 == "3.12-slim") {print $3}'
        # untagged leftovers of our builds, and of the old branch's, by their labels
        podman images -a --filter dangling=true --format '{{.ID}} {{json .Labels}}' |
            awk '/luna\.|ggml-org\/llama\.cpp|kokoro-reader/ {print $1}'
    } | sort -u)
fi

echo "Reset $ROOT to zero:"
echo "  in the checkout:  ${#ignored[@]} ignored files and folders (llama/models, .venv, configs, agent/ data...)"
[ "${#data[@]}" -gt 0 ] && echo "                    back as committed: ${data[*]}"
[ "${#outside[@]}" -gt 0 ] && printf '  outside it:       %s\n' "${outside[@]}"
if [ "$have_podman" = 1 ]; then
    echo "  containers:       ${containers[*]:-none}"
    echo "  volumes:          ${volumes[*]:-none}"
    echo "  secrets:          ${secrets[*]:-none}"
    [ "${#networks[@]}" -gt 0 ] && echo "  networks:         ${networks[*]}"
    echo "  images:           ${#images[@]} (ours, llama.cpp's, python:3.12-slim)"
fi
echo "Nothing is backed up."
if [ "$YES" != 1 ]; then
    read -r -p "Type reset to go on: " answer || answer=
    [ "$answer" = reset ] || { echo "Nothing changed."; exit 1; }
fi

# --- going ------------------------------------------------------------------------
if [ "$have_podman" = 1 ]; then
    [ "${#containers[@]}" -gt 0 ] && podman rm -f -t 5 "${containers[@]}" >/dev/null
    [ "${#volumes[@]}" -gt 0 ] && podman volume rm -f "${volumes[@]}" >/dev/null
    for s in "${secrets[@]}"; do podman secret rm "$s" >/dev/null; done
    for n in "${networks[@]}"; do podman network rm -f "$n" >/dev/null 2>&1 || true; done
    [ "${#images[@]}" -gt 0 ] && podman rmi -f "${images[@]}" >/dev/null 2>&1 || true
    rm -rf "${LUNA_RUN_DIR:-${XDG_RUNTIME_DIR:-/tmp}/luna-services-$(id -u)}"
    # The old isolated test's storage belongs to the containers' user IDs.
    [ -d "$HOME/.cache/luna-zero" ] && podman unshare rm -rf "$HOME/.cache/luna-zero"
fi
for p in "${outside[@]}"; do rm -rf "$p"; done
[ "${#data[@]}" -gt 0 ] && git checkout -q HEAD -- config.json agent
git clean -Xdfq
echo "At zero."

[ "$INSTALL" = 1 ] || { echo "Now: ./installer/install.sh   then: tests/zero/check.sh"; exit 0; }

# --- a new user's install, from here ------------------------------------------------
report="${XDG_CACHE_HOME:-$HOME/.cache}/luna-reset/last.txt"
mkdir -p "$(dirname "$report")"
args=(--no-system)
[ "$INTERACTIVE" = 1 ] || args+=(--yes)
start=$(date +%s)
echo
echo "== ./installer/install.sh ${args[*]}"
set +e
./installer/install.sh "${args[@]}"
installed=$?
PY=$(state venv_python)
[ -x "${PY:-}" ] || PY=.venv/bin/python
echo
echo "== python -m services up"
"$PY" -m services up --yes
echo
echo "== tests/zero/check.sh"
tests/zero/check.sh | tee "$report"
checked=${PIPESTATUS[0]}
set -e
echo
echo "Installer exit $installed; checks $([ "$checked" = 0 ] && echo passed || echo FAILED);" \
     "$(( ($(date +%s) - start) / 60 )) min. Report: $report"
[ "$installed" = 0 ] && [ "$checked" = 0 ]
