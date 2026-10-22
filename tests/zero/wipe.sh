#!/usr/bin/env bash
# This install, back to zero, for a first run as a new user would have it.
# Everything is backed up first; tests/zero/restore.sh puts it all back.
#
#   tests/zero/wipe.sh [BACKUP_DIR]     (default ~/.cache/luna-backup-<date-time>)
#
# Backed up, then removed:
#   config.json, install.json, llama/.api_key, llama/server.local.env,
#   services/services.local.env, deploy/.env, deploy/api_key, deploy/hf_token
#   the luna-models, luna-data and luna-tts-cache volumes (exported)
# Removed (they come back by themselves):
#   the luna-* containers, the luna_api_key and luna_hf_token secrets,
#   our images (luna-services-*, luna-deploy-*), the env-files
#
#   ZERO_YES=1      don't ask
#   ZERO_BASES=1    also upstream's base images (llama.cpp, python:3.12-slim), so they
#                   download again too - a true first run (default 1)
#   ZERO_VENV=1     also move Luna's Python environment (.venv) into the backup
#   ZERO_OLD=1      also remove luna-* images an older setup left (luna-llama, luna-tts)
#
# Your models folder (llama/models, LM Studio's), your Hugging Face token,
# agent/ (her memory) and the git checkout are never touched.
set -euo pipefail
cd "$(dirname "$(realpath "$0")")/../.."
command -v podman >/dev/null || { echo "This needs Podman." >&2; exit 1; }

BACKUP="${1:-$HOME/.cache/luna-backup-$(date +%Y%m%d-%H%M%S)}"
FILES=(config.json install.json llama/.api_key llama/server.local.env services/services.local.env
       deploy/.env deploy/api_key deploy/hf_token)
VOLUMES=(luna-models luna-data luna-tts-cache)
SECRETS=(luna_api_key luna_hf_token)

mapfile -t containers < <(podman ps -a --filter name=^luna- --format '{{.Names}}' | grep -v '^luna-test-' || true)
mapfile -t images < <( { podman images -q --filter label=luna.kind
    podman images --format '{{.Repository}}:{{.Tag}} {{.ID}}' | awk '/^localhost\/luna-(services|deploy)-/ {print $2}'
    if [ "${ZERO_BASES:-1}" = 1 ]; then
        podman images --format '{{.Repository}} {{.ID}}' |
            awk '$1 == "ghcr.io/ggml-org/llama.cpp" || $1 == "docker.io/library/python" {print $2}'
    fi
    if [ "${ZERO_OLD:-0}" = 1 ]; then
        podman images --format '{{.Repository}} {{.ID}}' |
            awk '$1 == "localhost/luna-llama" || $1 == "localhost/luna-tts" {print $2}'
    fi; } | sort -u)
present_files=(); for f in "${FILES[@]}"; do [ -e "$f" ] && present_files+=("$f"); done
present_vols=(); for v in "${VOLUMES[@]}"; do podman volume exists "$v" && present_vols+=("$v"); done

echo "Back to zero: $(pwd)"
echo "  back up to $BACKUP, then remove:"
echo "    files:      ${present_files[*]:-none}"
echo "    volumes:    ${present_vols[*]:-none}  (exported first)"
echo "    containers: ${containers[*]:-none}"
echo "    images:     ${#images[@]}"
echo "    secrets:    ${SECRETS[*]}"
[ "${ZERO_VENV:-0}" = 1 ] && [ -d .venv ] && echo "    .venv       (moved into the backup)"
if [ "${ZERO_YES:-0}" != 1 ]; then
    read -r -p "Go on? [y/N] " yn || yn=n
    case "$yn" in [Yy]*) ;; *) echo "Nothing changed."; exit 1 ;; esac
fi

mkdir -p "$BACKUP/files" "$BACKUP/volumes"
chmod 700 "$BACKUP"
for f in "${present_files[@]}"; do
    cp -a --parents "$f" "$BACKUP/files/"
done
for v in "${present_vols[@]}"; do
    echo "  exporting $v"
    podman volume export "$v" -o "$BACKUP/volumes/$v.tar"
done
{ echo "repo: $(pwd)"; echo "date: $(date)"; echo "containers: ${containers[*]:-}"
  echo "files: ${present_files[*]:-}"; echo "volumes: ${present_vols[*]:-}"; } > "$BACKUP/what.txt"

[ "${#containers[@]}" -gt 0 ] && podman rm -f -t 10 "${containers[@]}" >/dev/null
for v in "${present_vols[@]}"; do podman volume rm -f "$v" >/dev/null; done
for s in "${SECRETS[@]}"; do podman secret rm "$s" >/dev/null 2>&1 || true; done
[ "${#images[@]}" -gt 0 ] && podman rmi -f "${images[@]}" >/dev/null 2>&1 || true
rm -rf "${LUNA_RUN_DIR:-${XDG_RUNTIME_DIR:-/tmp}/luna-services-$(id -u)}"
rm -f "${present_files[@]}"
if [ "${ZERO_VENV:-0}" = 1 ] && [ -d .venv ]; then
    mv .venv "$BACKUP/venv"
fi

echo "Done - backed up in $BACKUP."
echo "Now, as a new user would:  ./installer/install.sh"
echo "Check it:                  tests/zero/check.sh"
echo "Put yours back:            tests/zero/restore.sh $BACKUP"
