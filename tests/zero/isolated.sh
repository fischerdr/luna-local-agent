#!/usr/bin/env bash
# Luna's containers installed from zero, beside your own setup: a fresh
# clone of this branch, a HOME of its own (so no venv, pip cache, Hugging
# Face token or launcher of yours) and a Podman storage of its own (its
# own images, volumes and secrets). Everything downloads, as for a new
# user. Nothing of yours is read or changed - except that your luna-*
# containers are stopped while it runs, since it uses the same ports, and
# started again after.
#
#   tests/zero/isolated.sh           install from zero, check it, stop it
#   tests/zero/isolated.sh --keep    leave the clean install running (yours stay stopped)
#   tests/zero/isolated.sh --clean   remove all of it
#
#   ZERO_DIR=<dir>             where it lives (default ~/.cache/luna-zero; ~30 GB)
#   ZERO_ARGS="..."            installer flags (default: --yes --no-system, a new
#                              user's defaults; drop --yes to answer it yourself)
#   ZERO_WIPE_TEST=1           afterwards, also run wipe.sh and restore.sh on the
#                              clean install, to test those two
#
# The installer runs with --no-system so it never needs sudo; the system
# packages are this machine's. The clone is of what's committed - commit
# first to test a change.
set -euo pipefail

REPO="$(cd "$(dirname "$(realpath "$0")")/../.." && pwd)"
ZERO="${ZERO_DIR:-$HOME/.cache/luna-zero}"
REAL_PODMAN="$(command -v podman)" || { echo "This needs Podman." >&2; exit 1; }
KEEP=0

zpodman() {
    "$REAL_PODMAN" --root "$ZERO/storage" --runroot "$ZERO/run" --tmpdir "$ZERO/tmp" "$@"
}

clean() {
    [ -d "$ZERO" ] || return 0
    echo "Removing the clean install in $ZERO"
    if [ -d "$ZERO/storage" ]; then
        zpodman rm -af -t 2 >/dev/null 2>&1 || true
        zpodman system reset -f >/dev/null 2>&1 || true
    fi
    # Files in it belong to the container's users: remove them from inside
    # Podman's user namespace.
    "$REAL_PODMAN" unshare rm -rf "$ZERO"
}

case "${1:-}" in
--clean) clean; exit 0 ;;
--keep) KEEP=1 ;;
"") ;;
*) echo "usage: $0 [--keep|--clean]" >&2; exit 2 ;;
esac

free=$(df -Pk "$(dirname "$ZERO")" | awk 'NR==2 {print int($4 / 1048576)}')
if [ "$free" -lt 30 ]; then
    echo "Only $free GB free where $ZERO goes; a clean install needs about 30 GB (ZERO_DIR moves it)." >&2
    exit 1
fi

# --- your containers: stopped for the ports, started again after -----------
mapfile -t yours < <("$REAL_PODMAN" ps --filter name=^luna- --format '{{.Names}}' | grep -v '^luna-test-' || true)
restart_yours() {
    if [ "$KEEP" = 1 ]; then
        [ "${#yours[@]}" -gt 0 ] && echo "Yours are stopped: $REAL_PODMAN start ${yours[*]}"
        return 0
    fi
    # Only the clean install lives in its storage: stop all of it.
    [ -d "$ZERO/storage" ] && zpodman rm -af -t 5 >/dev/null 2>&1
    if [ "${#yours[@]}" -gt 0 ]; then
        echo "Starting yours again: ${yours[*]}"
        "$REAL_PODMAN" start "${yours[@]}" >/dev/null
    fi
}
trap restart_yours EXIT

if [ "${#yours[@]}" -gt 0 ]; then
    echo "Your running containers use the same ports: ${yours[*]}"
    read -r -p "Stop them while the clean install runs (started again after)? [Y/n] " yn || yn=y
    case "${yn:-y}" in [Yy]*) ;; *) yours=(); echo "Left running - the clean install will find its ports taken."; exit 1 ;; esac
    "$REAL_PODMAN" stop -t 10 "${yours[@]}" >/dev/null
fi
for port in 8080 8081 8082 8899; do
    if ss -ltnH "sport = :$port" 2>/dev/null | grep -q .; then
        echo "Port $port is still taken (a host llama/start.sh, LM Studio?) - stop it first." >&2
        exit 1
    fi
done

# --- a clean world ------------------------------------------------------------
clean
mkdir -p "$ZERO"/{home,storage,run,tmp,bin,runtime}
chmod 700 "$ZERO/runtime"
# Every podman call in the clean install - the installer's, python -m
# services', podman-compose's - goes to its own storage.
cat > "$ZERO/bin/podman" <<EOF
#!/usr/bin/env bash
exec "$REAL_PODMAN" --root "$ZERO/storage" --runroot "$ZERO/run" --tmpdir "$ZERO/tmp" "\$@"
EOF
chmod +x "$ZERO/bin/podman"

branch=$(git -C "$REPO" rev-parse --abbrev-ref HEAD)
echo "Cloning $branch ($(git -C "$REPO" rev-parse --short HEAD)) into $ZERO/repo"
git clone --quiet --no-hardlinks --branch "$branch" "$REPO" "$ZERO/repo"
[ -z "$(git -C "$REPO" status --porcelain --untracked-files=no)" ] ||
    echo "  (your uncommitted changes aren't in it)"

# The clean install's surroundings: its own HOME and env-file dir, your
# runtime dir (Podman's systemd timers for health checks need it).
zenv=(env -i "HOME=$ZERO/home" "PATH=$ZERO/bin:/usr/local/bin:/usr/bin:/bin"
      "XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}" "LUNA_RUN_DIR=$ZERO/runtime"
      "TERM=${TERM:-xterm}" "LANG=${LANG:-C.UTF-8}"
      "DBUS_SESSION_BUS_ADDRESS=${DBUS_SESSION_BUS_ADDRESS:-}")
read -r -a args <<< "${ZERO_ARGS:---yes --no-system}"

log="$ZERO/install.log"
echo
echo "== Installing from zero: ./installer/install.sh ${args[*]}   (log: $log)"
start=$(date +%s)
( cd "$ZERO/repo" && "${zenv[@]}" ./installer/install.sh "${args[@]}" ) 2>&1 | tee "$log"
installed=${PIPESTATUS[0]}

py=$(cd "$ZERO/repo" && python3 -c 'import json; print(json.load(open("install.json")).get("venv_python", ""))' 2>/dev/null || true)
[ -x "${py:-}" ] || py="$ZERO/repo/.venv/bin/python"

echo
echo "== python -m services up   (the first one downloads the models)"
( cd "$ZERO/repo" && "${zenv[@]}" "$py" -m services up ) 2>&1 | tee -a "$log"

echo
echo "== Checking it"
set +e
( cd "$ZERO/repo" && "${zenv[@]}" tests/zero/check.sh ) 2>&1 | tee "$ZERO/report.txt"
checked=${PIPESTATUS[0]}

if [ "${ZERO_WIPE_TEST:-0}" = 1 ]; then
    echo
    echo "== wipe.sh and restore.sh, on the clean install"
    ( cd "$ZERO/repo" && "${zenv[@]}" tests/zero/selftest-wipe.sh ) 2>&1 | tee -a "$ZERO/report.txt"
    [ "${PIPESTATUS[0]}" = 0 ] || checked=1
fi
set -e

minutes=$(( ($(date +%s) - start) / 60 ))
echo
echo "Installer exit $installed; checks $([ "$checked" = 0 ] && echo passed || echo FAILED); ${minutes} min."
echo "Report: $ZERO/report.txt   Install log: $log"
echo "Look around: cd $ZERO/repo   (podman for it: $ZERO/bin/podman)   Remove it: $0 --clean"
[ "$installed" = 0 ] && [ "$checked" = 0 ]
