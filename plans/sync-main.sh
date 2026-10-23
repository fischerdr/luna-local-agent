#!/usr/bin/env bash
# Bring the owner's latest main into every branch of ours, in order, and
# test the result. Stops at the first conflict or failing test, and says
# which branch.
#
#   plans/sync-main.sh            fetch, merge, test (nothing is pushed)
#   plans/sync-main.sh --live     also tests/zero/check.sh against the running containers
#   plans/sync-main.sh --push     also push every branch to origin (your fork) when all is good
#
# The order (no rebase, ever):
#   1. upstream/main  -> main                  fast-forward only: main stays the owner's
#   2. main           -> feat/plugin-hooks     PR 1
#      feat/plugin-hooks -> feat/containers-plugin   PR 2
#      main           -> each fix/*            (says when main has the fix already)
#      main           -> feat/containers       the reference branch
#      feat/containers-plugin, fix/* -> test/containers   what Luna runs from
#   3. tests: the core suite on feat/plugin-hooks; the core and backend
#      suites on test/containers (and check.sh with --live)
#
# Lives on feat/containers only, so it never lands in a PR. It copies
# itself to a temp file first: switching branches takes this file away.
set -uo pipefail

if [ -z "${SYNC_MAIN_COPY:-}" ]; then
    copy=$(mktemp "${TMPDIR:-/tmp}/sync-main.XXXXXX")
    cp "$0" "$copy"
    SYNC_MAIN_COPY=$copy exec bash "$copy" "$@"
fi
trap 'rm -f "$SYNC_MAIN_COPY"' EXIT

LIVE=0 PUSH=0
for a in "$@"; do
    case $a in
        --live) LIVE=1 ;;
        --push) PUSH=1 ;;
        -h|--help) sed -n '2,22p' "$SYNC_MAIN_COPY" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $a (see --help)"; exit 2 ;;
    esac
done

ROOT=$(git rev-parse --show-toplevel) || exit 1
cd "$ROOT" || exit 1

HOOKS=feat/plugin-hooks
PLUGIN=feat/containers-plugin
REFERENCE=feat/containers
TEST=test/containers
# Your runtime files: carried across every switch, never committed.
RUNTIME=(agent/agent.json config.json)

say()  { printf '\n== %s\n' "$*"; }
ok()   { printf '  ✓ %s\n' "$*"; }
note() { printf '  · %s\n' "$*"; }
stop() { printf '\n  ✗ %s\n' "$*"; printf '  (stopped on %s)\n' "$(git branch --show-current)"; exit 1; }

# --- before anything: a tree with nothing but the runtime files changed -----
start=$(git branch --show-current)
dirty=$(git status --porcelain --untracked-files=no | awk '{print $2}' | grep -vxF -f <(printf '%s\n' "${RUNTIME[@]}"))
[ -z "$dirty" ] || stop "uncommitted changes besides ${RUNTIME[*]}: $(echo $dirty) - commit or stash them first"
git remote get-url upstream >/dev/null 2>&1 || stop "no 'upstream' remote (the owner's repo)"
for b in main "$HOOKS" "$PLUGIN" "$REFERENCE" "$TEST"; do
    git rev-parse -q --verify "refs/heads/$b" >/dev/null || stop "no branch $b here"
done
mapfile -t FIXES < <(git for-each-ref --format='%(refname:short)' 'refs/heads/fix/*')

PY=$(python3 -c 'import json; print(json.load(open("install.json")).get("venv_python", ""))' 2>/dev/null)
[ -x "${PY:-}" ] || PY=.venv/bin/python
[ -x "$PY" ] || stop "no Python environment (.venv) to test with"

# --- 1. upstream/main -> main ------------------------------------------------
say "1. The owner's main"
git fetch -q upstream || stop "couldn't fetch upstream"
before=$(git rev-parse main)
if [ "$(git branch --show-current)" = main ]; then
    git merge -q --ff-only upstream/main || stop "main has commits upstream doesn't - it should only ever follow upstream"
else
    git fetch -q . upstream/main:main || stop "main has commits upstream doesn't - it should only ever follow upstream"
fi
if [ "$before" = "$(git rev-parse main)" ]; then
    ok "main is already upstream's ($(git log -1 --format='%h %s' main))"
else
    ok "main: $(git rev-list --count "$before..main") new commit(s), now $(git log -1 --format='%h %s' main)"
    git log --format='      %h %s' "$before..main" | head -20
fi

# --- 2. merge forward ----------------------------------------------------------
# merge INTO FROM: switch to INTO and merge FROM, stopping on a conflict.
merge() {
    local into=$1 from=$2
    git switch -q "$into" || stop "couldn't switch to $into"
    if git merge-base --is-ancestor "$from" HEAD; then
        note "$into already has $from"
        return
    fi
    if git merge -q --no-ff --no-edit "$from" >/dev/null 2>&1; then
        ok "$from -> $into"
    else
        echo
        git diff --name-only --diff-filter=U | sed 's/^/      conflict: /'
        stop "$from -> $into: merge conflict. Fix the files above, git add them, git commit, then run this again"
    fi
}

say "2. Merging forward"
merge "$HOOKS" main
merge "$PLUGIN" "$HOOKS"
for f in "${FIXES[@]}"; do
    merge "$f" main
    # Only its own PR text left: main has this fix by now.
    if git diff --quiet main "$f" -- . ':(exclude)PR-*.md'; then
        note "$f: no change from main any more - main has this fix; delete the branch (git branch -d $f)"
    fi
done
merge "$REFERENCE" main
merge "$TEST" "$PLUGIN"
for f in "${FIXES[@]}"; do
    merge "$TEST" "$f"
done

# --- 3. tests ----------------------------------------------------------------------
LOG=$(mktemp -d "${TMPDIR:-/tmp}/sync-main-tests.XXXXXX")

# suite NAME ARGS...: one unittest run, its summary line, and stop if it fails.
suite() {
    local name=$1; shift
    local out="$LOG/$(echo "$(git branch --show-current)-$name" | tr '/ ' '--').txt"
    "$PY" -m unittest "$@" >"$out" 2>&1
    local code=$?
    # a test on main makes tools/askpass.sh executable; never leave that behind
    git checkout -q -- tools/askpass.sh 2>/dev/null
    local result
    result=$(grep -E '^(Ran|OK|FAILED)' "$out" | tr '\n' ' ')
    if [ "$code" -ne 0 ]; then
        grep -E '^(ERROR|FAIL):' "$out" | head -10 | sed 's/^/      /'
        stop "$name tests fail on $(git branch --show-current): $result- the whole output is in $out"
    fi
    ok "$name: $result"
}

say "3. Tests"
git switch -q "$HOOKS"
suite "core" discover tests
git switch -q "$TEST"
suite "core" discover tests
suite "backend" discover -s backends/containers/tests -t .
if [ "$LIVE" = 1 ]; then
    if backends/containers/tests/zero/check.sh >"$LOG/check.txt" 2>&1; then
        ok "check.sh: all good"
    else
        grep '✗' "$LOG/check.txt" | head -10 | sed 's/^/      /'
        stop "check.sh fails on $TEST - the whole output is in $LOG/check.txt"
    fi
else
    note "check.sh skipped (--live runs it against the running containers)"
fi

# --- push, if asked -----------------------------------------------------------------
if [ "$PUSH" = 1 ]; then
    say "Pushing to origin (your fork)"
    git push -q origin main "$HOOKS" "$PLUGIN" "${FIXES[@]}" "$REFERENCE" "$TEST" >"$LOG/push.txt" 2>&1 ||
        { grep -v '^remote:' "$LOG/push.txt" | sed 's/^/      /'; stop "the push to origin failed (above)"; }
    ok "pushed: main $HOOKS $PLUGIN ${FIXES[*]} $REFERENCE $TEST"
else
    note "nothing pushed - run again with --push, or: git push origin main $HOOKS $PLUGIN ${FIXES[*]} $REFERENCE $TEST"
fi

git switch -q "$TEST"
printf '\nAll good. On %s (you started on %s).\n' "$TEST" "${start:-a detached HEAD}"
