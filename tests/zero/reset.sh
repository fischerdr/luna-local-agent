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
#   tests/zero/reset.sh --yes ...         don't ask about the checkout
#   tests/zero/reset.sh --outside ...     yes to every outside-the-checkout group too
#
# Inside the checkout it asks once. Everything outside it is asked about
# group by group (default no), unless --outside says yes to all of them -
# and --install asks before the installer writes outside it too.
# System packages are never touched; the installer runs with --no-system.
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

YES=0 INSTALL=0 INTERACTIVE=0 OUTSIDE=0
for a in "$@"; do
    case "$a" in
    --yes) YES=1 ;;
    --outside) OUTSIDE=1 ;;
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
data_dirs=() launchers=() venvs=()
[ -d "${XDG_CACHE_HOME:-$HOME/.cache}/ai-voice" ] && data_dirs+=("${XDG_CACHE_HOME:-$HOME/.cache}/ai-voice")
[ -d "$HOME/.cache/luna-zero" ] && data_dirs+=("$HOME/.cache/luna-zero")
[ "$(readlink -f "$HOME/.local/bin/luna" 2>/dev/null)" = "$ROOT/start.sh" ] && launchers+=("$HOME/.local/bin/luna")
desktop="$HOME/.local/share/applications/luna.desktop"
[ -f "$desktop" ] && grep -q "$ROOT" "$desktop" && launchers+=("$desktop")
case "$venv" in "$ROOT"/*|"") ;; *) [ -d "$venv" ] && venvs+=("$venv") ;; esac
[ -n "$kokoro" ] && [ -d "$kokoro/server/.venv" ] && venvs+=("$kokoro/server/.venv")

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

# ask GROUP-DESCRIPTION: yes when --outside, else a y/N question (default no)
ask() {
    if [ "$OUTSIDE" = 1 ]; then echo "  $1 yes (--outside)"; return 0; fi
    local yn
    read -r -p "  $1 [y/N] " yn || yn=n
    case "$yn" in [Yy]*) return 0 ;; *) return 1 ;; esac
}

echo "Reset $ROOT to zero (nothing is backed up)."
echo "In the checkout: ${#ignored[@]} gitignored files and folders (.venv, install state, keys,"
echo "  local env files, llama/models, agent/'s databases, __pycache__)"
[ "${#data[@]}" -gt 0 ] && echo "  and back as committed: ${data[*]}"
if [ "$YES" != 1 ]; then
    read -r -p "Type reset to go on: " answer || answer=
    [ "$answer" = reset ] || { echo "Nothing changed."; exit 1; }
fi

echo "Outside the checkout - each only if you say so:"
gone_outside=()
if [ "${#data_dirs[@]}" -gt 0 ] && ask "her logs and knowledge index, and old test data: ${data_dirs[*]}?"; then
    for p in "${data_dirs[@]}"; do
        # the old isolated test's storage belongs to the containers' user IDs
        if [ "$p" = "$HOME/.cache/luna-zero" ] && [ "$have_podman" = 1 ]; then podman unshare rm -rf "$p"; else rm -rf "$p"; fi
    done
    gone_outside+=("${data_dirs[@]}")
fi
if [ "${#launchers[@]}" -gt 0 ] && ask "the launchers: ${launchers[*]}?"; then
    rm -f "${launchers[@]}"; gone_outside+=("${launchers[@]}")
fi
if [ "${#venvs[@]}" -gt 0 ] && ask "Python environments the installer made: ${venvs[*]}?"; then
    rm -rf "${venvs[@]}"; gone_outside+=("${venvs[@]}")
fi
if [ "$have_podman" = 1 ]; then
    if [ "${#containers[@]}${#volumes[@]}${#secrets[@]}${#networks[@]}" != 0000 ] &&
       ask "Podman: containers (${containers[*]:-none}), volumes with the models (${volumes[*]:-none}), secrets (${secrets[*]:-none})${networks:+, networks (${networks[*]})}?"; then
        [ "${#containers[@]}" -gt 0 ] && podman rm -f -t 5 "${containers[@]}" >/dev/null
        [ "${#volumes[@]}" -gt 0 ] && podman volume rm -f "${volumes[@]}" >/dev/null
        for x in "${secrets[@]}"; do podman secret rm "$x" >/dev/null; done
        for x in "${networks[@]}"; do podman network rm -f "$x" >/dev/null 2>&1 || true; done
        rm -rf "${LUNA_RUN_DIR:-${XDG_RUNTIME_DIR:-/tmp}/luna-services-$(id -u)}"
        gone_outside+=("Podman containers, volumes, secrets")
    fi
    if [ "${#images[@]}" -gt 0 ] && ask "Podman images: ${#images[@]} (ours, upstream's llama.cpp ones, python:3.12-slim)?"; then
        podman rmi -f "${images[@]}" >/dev/null 2>&1 || true
        gone_outside+=("${#images[@]} images")
    fi
fi

[ "${#data[@]}" -gt 0 ] && git checkout -q HEAD -- config.json agent
# On NFS a file a running program still has open (an editor's Python tools
# in .venv, say) stays behind as .nfs* until that program lets go: say
# who, and carry on - the install recreates around it.
if ! git clean -Xdfq 2>/dev/null; then
    mapfile -t left < <(git clean -Xdn | sed 's/^Would remove //')
    others=$(find "${left[@]}" -type f ! -name '.nfs*' 2>/dev/null | head -5)
    if [ -n "$others" ]; then
        echo "Couldn't remove: $others" >&2
        exit 1
    fi
    echo "  still held open by running programs (NFS keeps them until they exit):"
    find "${left[@]}" -type f -name '.nfs*' 2>/dev/null | while read -r f; do
        for pid in $(fuser "$f" 2>/dev/null); do
            echo "    $(ps -o args= -p "$pid" | cut -c1-120)"
        done
    done | sort -u
    echo "  (an editor's Python tools? restart them once the install has made .venv again)"
fi
echo "The checkout is at zero.${gone_outside:+ Outside it, gone: ${gone_outside[*]}.}"

[ "$INSTALL" = 1 ] || { echo "Now: ./installer/install.sh   then: tests/zero/check.sh"; exit 0; }

echo
echo "The installer writes outside the checkout too, as for a new user: the luna launcher"
echo "(~/.local/bin, ~/.local/share/applications), pip's cache, Podman images, volumes and secrets."
ask "Run it?" || { echo "Not installing. When you want: ./installer/install.sh"; exit 0; }

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
