#!/usr/bin/env bash
# wipe.sh and restore.sh tested on an install (isolated.sh runs this on its
# clean one, with ZERO_WIPE_TEST=1): wipe leaves nothing of ours behind,
# restore brings it all back, and check.sh passes again.
set -uo pipefail
cd "$(dirname "$(realpath "$0")")/../.." || exit 1
BACKUP=$(mktemp -d)/backup
bad=0
ok() { echo "  ✓ $*"; }
no() { echo "  ✗ $*"; bad=$((bad + 1)); }
check() { local label=$1; shift; if "$@"; then ok "$label"; else no "$label"; fi; }

key_before=$(cat llama/.api_key)
ZERO_YES=1 ZERO_BASES=0 tests/zero/wipe.sh "$BACKUP" || no "wipe.sh failed"

left=$(podman ps -a --format '{{.Names}}' | grep '^luna-' | grep -v '^luna-test-')
check "no luna-* containers left ${left}" test -z "$left"
vols=$(podman volume ls --format '{{.Name}}' | grep -E '^luna-(models|data|tts-cache)$')
check "no volumes left ${vols}" test -z "$vols"
secrets=$(podman secret ls --format '{{.Name}}' | grep -E '^luna_(api_key|hf_token)$')
check "no secrets left ${secrets}" test -z "$secrets"
imgs=$(podman images -q --filter label=luna.kind)
check "none of our images left ${imgs}" test -z "$imgs"
for f in config.json install.json llama/.api_key services/services.local.env; do
    check "$f gone" test ! -e "$f"
done
check "luna-models exported" test -f "$BACKUP/volumes/luna-models.tar"

ZERO_YES=1 tests/zero/restore.sh "$BACKUP" || no "restore.sh failed"
check "the same key is back" test "$(cat llama/.api_key 2>/dev/null)" = "$key_before"
check "check.sh passes after the restore" tests/zero/check.sh
rm -rf "$(dirname "$BACKUP")"

[ "$bad" = 0 ] && echo "wipe and restore: all good." || echo "wipe and restore: $bad failed."
[ "$bad" = 0 ]
