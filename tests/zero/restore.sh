#!/usr/bin/env bash
# Put back what tests/zero/wipe.sh backed up: the files, the volumes, the
# Python environment if it was moved - then the images and containers,
# from those settings. Whatever a fresh install made since is replaced.
#
#   tests/zero/restore.sh BACKUP_DIR
#   ZERO_YES=1    don't ask
set -euo pipefail
cd "$(dirname "$(realpath "$0")")/../.."
BACKUP="${1:?usage: $0 BACKUP_DIR (wipe.sh printed it)}"
[ -f "$BACKUP/what.txt" ] || { echo "$BACKUP isn't a wipe.sh backup." >&2; exit 1; }
command -v podman >/dev/null || { echo "This needs Podman." >&2; exit 1; }

echo "Restore $(pwd) from $BACKUP:"
sed 's/^/    /' "$BACKUP/what.txt"
if [ "${ZERO_YES:-0}" != 1 ]; then
    read -r -p "Replace what's here now with that? [y/N] " yn || yn=n
    case "$yn" in [Yy]*) ;; *) echo "Nothing changed."; exit 1 ;; esac
fi

# What the fresh install made: its containers, volumes and secrets.
mapfile -t containers < <(podman ps -a --filter name=^luna- --format '{{.Names}}' | grep -v '^luna-test-' || true)
[ "${#containers[@]}" -gt 0 ] && podman rm -f -t 10 "${containers[@]}" >/dev/null
for v in luna-models luna-data luna-tts-cache; do podman volume rm -f "$v" >/dev/null 2>&1 || true; done
for s in luna_api_key luna_hf_token; do podman secret rm "$s" >/dev/null 2>&1 || true; done

# Files, as they were (a fresh install's own copies go).
for f in config.json install.json llama/.api_key llama/server.local.env services/services.local.env \
         deploy/.env deploy/api_key deploy/hf_token; do
    rm -f "$f"
done
cp -a "$BACKUP/files/." .
for tar in "$BACKUP"/volumes/*.tar; do
    [ -e "$tar" ] || continue
    v=$(basename "$tar" .tar)
    echo "  importing $v"
    podman volume create "$v" >/dev/null
    podman volume import "$v" "$tar"
done
if [ -d "$BACKUP/venv" ]; then
    rm -rf .venv
    mv "$BACKUP/venv" .venv
fi

PY=$(python3 -c 'import json; print(json.load(open("install.json")).get("venv_python", ""))' 2>/dev/null || true)
[ -x "${PY:-}" ] || PY=.venv/bin/python
if [ ! -x "$PY" ]; then
    echo "Files and volumes are back; no Python environment yet - run ./installer/install.sh."
    exit 0
fi

# The images these settings use, if a wipe took them; then the containers.
"$PY" - <<'EOF'
import subprocess
from services import images, lifecycle, settings
s = settings.load()
backend, _ = images.image_for_host(s["LLAMA_BACKEND"])
voice = images.tts_backend(backend) if lifecycle.tts_device(s, backend) == "cuda" else "cpu"
for kind, b in sorted({("llama", backend), ("llama", "cpu"), ("tts", voice)}):
    if not images.find(b, kind=kind):
        print(f"  building {images.tag(b, kind)}", flush=True)
        subprocess.run(images.build_cmd(b, kind), check=True, stdout=subprocess.DEVNULL)
EOF
"$PY" -m services up --yes
echo "Restored. Check it: tests/zero/check.sh"
