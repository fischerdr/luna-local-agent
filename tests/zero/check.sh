#!/usr/bin/env bash
# After an install from zero: is everything there, and does it answer?
# It changes nothing but a cache: Luna's recall check leaves three test
# facts' vectors in agent/embeddings.db. Run it from the repo to check:
#
#   tests/zero/check.sh
#
# Each line is ✓ or ✗; it exits 1 if any is ✗. isolated.sh runs it in
# its own clean world; after wipe.sh and a fresh install, run it yourself.
set -uo pipefail
cd "$(dirname "$(realpath "$0")")/../.." || exit 1

bad=0
ok()   { echo "  ✓ $*"; }
no()   { echo "  ✗ $*"; bad=$((bad + 1)); }
# check LABEL COMMAND...: ✓ when the command succeeds
check() { local label=$1; shift; if "$@"; then ok "$label"; else no "$label"; fi; }
quiet() { "$@" >/dev/null 2>&1; }          # the command's own output only; check still says ✓/✗
key_private() { [ -s llama/.api_key ] && [ "$(stat -c %a llama/.api_key)" = 600 ]; }

# Luna's own Python, as the installer recorded it.
PY=$(python3 -c 'import json; print(json.load(open("install.json")).get("venv_python", ""))' 2>/dev/null)
[ -x "${PY:-}" ] || PY=.venv/bin/python
[ -x "$PY" ] || { echo "No Python environment here - the install didn't get that far."; exit 1; }
echo "Checking $(pwd) with $PY"

echo "Setup"
check "llama/.api_key, readable only by you" key_private
check "config.json: services.autostart" grep -q '"autostart": true' config.json
check "install.json: the voice is the container" python3 -c \
    'import json, sys; v = json.load(open("install.json")).get("voice", {}); sys.exit(v.get("container") != "luna-tts")' 2>/dev/null

echo "Containers"
status=$("$PY" -m services status 2>&1)
while IFS= read -r ln; do echo "    $ln"; done <<< "$status"
check "every role ready" test "$(grep -c " · " <<< "$status")" = 0
listen=$(ss -ltnH 2>/dev/null | awk '{print $4}' | grep -E ':(8080|8081|8082|8899)$')
if [ -n "$listen" ] && ! echo "$listen" | grep -qv '^127\.0\.0\.1:'; then
    ok "listening on 127.0.0.1 only: $(echo "$listen" | tr '\n' ' ')"
else
    no "listening: ${listen:-nothing} (want 127.0.0.1:8080-8082 and :8899)"
fi

echo "Checks"
check "./start.sh --check" quiet ./start.sh --check
health=$("$PY" installer/install.py --check 2>&1)
check "installer --check" test "$(grep -c "✗" <<< "$health")" = 0

echo "Each server, with the key"
KEY=$(cat llama/.api_key 2>/dev/null)
auth=(-H "Authorization: Bearer $KEY" -H "Content-Type: application/json")
code=$(curl -s -o /dev/null -w '%{http_code}' -X POST http://127.0.0.1:8080/v1/chat/completions \
    -H "Content-Type: application/json" -d '{}')
check "chat refuses without the key ($code)" test "$code" = 401
reply=$(curl -s --max-time 300 "${auth[@]}" http://127.0.0.1:8080/v1/chat/completions \
    -d '{"messages":[{"role":"user","content":"Say hello in five words."}],"max_tokens":32,"reasoning_budget_tokens":0}')
check "chat answers" python3 -c \
    'import json,sys; m=json.loads(sys.argv[1])["choices"][0]["message"]; sys.exit(not (m.get("content") or "").strip())' "$reply" 2>/dev/null
dim=$(curl -s --max-time 60 "${auth[@]}" http://127.0.0.1:8081/v1/embeddings -d '{"input":"hello"}' |
    python3 -c 'import json,sys; print(len(json.load(sys.stdin)["data"][0]["embedding"]))' 2>/dev/null)
check "embeddings (${dim:-none} wide)" test "${dim:-0}" -gt 0
best=$(curl -s --max-time 60 "${auth[@]}" http://127.0.0.1:8082/v1/rerank \
    -d '{"query":"what pet does the user have","documents":["The user likes jazz.","The user has a cat named Miso."]}' |
    python3 -c 'import json,sys; r=json.load(sys.stdin)["results"]; print(max(r, key=lambda x: x["relevance_score"])["index"])' 2>/dev/null)
check "the reranker puts the answer first" test "${best:-}" = 1
wav=$(curl -s --max-time 120 -X POST http://127.0.0.1:8899/tts -H "Content-Type: application/json" \
    -d '{"text":"Hello, I am Luna.","voice":"af_bella"}' | head -c 4)
check "her voice speaks (a WAV)" test "$wav" = RIFF

echo "Luna's own code"
probe=$("$PY" - <<'EOF' 2>&1
import config, embedmem, speech
facts = ["The user likes jazz.", "The user has a cat named Miso.", "Paris is in France."]
ranked = embedmem.rank("what pet does the user have", facts) or []
samples, rate = speech._synthesize("Hello from a clean install.")
print("PROBE", config.LLM_BACKEND, embedmem.available(), embedmem.reranker_available(),
      ranked[0][1] if ranked else None, len(samples) > rate // 2)
EOF
)
line=$(echo "$probe" | grep '^PROBE')
check "picks llama.cpp, recalls and reranks by meaning, speaks (${line:-$(tail -1 <<< "$probe")})" \
    test "$line" = "PROBE llama True True The user has a cat named Miso. True"

echo "Room"
if command -v nvidia-smi >/dev/null 2>&1; then
    mem=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | head -1)
    used=${mem%%,*}; total=${mem##*, }
    check "VRAM ${used} of ${total} MiB" test "$used" -lt "$total"
fi
log="${XDG_CACHE_HOME:-$HOME/.cache}/ai-voice/ai-voice.log"
if [ -f "$log" ]; then
    n=$(grep -ciE 'embed.*(timed out|timeout)' "$log")
    check "no embed timeouts in $log ($n)" test "$n" = 0
fi

echo
if [ "$bad" = 0 ]; then echo "All good."; else echo "$bad check(s) failed."; fi
[ "$bad" = 0 ]
