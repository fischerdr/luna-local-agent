"""The embedder and reranker on demand: started when a turn wants them,
stopped when they've been idle a while. Off unless EMBED_ONDEMAND=on in
services/services.local.env.

embedmem.py tells wanted() each time it asks whether a server is up. When
one is down, its container is started in a thread - the turn doesn't
wait; it falls back as it always does (word matching, the embedding
order), and the next turn finds the server. A reaper stops what this
started once nobody has wanted it for EMBED_IDLE (or RERANK_IDLE)
seconds. Containers started any other way are never stopped here.

Only for servers on this machine (embed_url and rerank_url on
127.0.0.1/localhost), and only on Linux with Podman.
"""
import shutil
import sys
import threading
import time
from urllib.parse import urlparse

from services import keys, lifecycle, settings

ROLE_OF = {"embed": ("EMBED_IDLE", 600), "rerank": ("RERANK_IDLE", 600)}
REAP_EVERY = 30

_lock = threading.Lock()     # guards the sets below; never held while podman runs
_starting = set()
_started = set()
_last = {}
_reaper = None
_settings = None


def _load():
    global _settings
    if _settings is None:
        try:
            _settings = settings.load()
        except settings.SettingsError:
            _settings = {}
    return _settings


def enabled():
    """Whether on-demand is switched on, and can work here. Never in a test
    run: tests/offline.py makes every server look down, and that mustn't
    start your real containers."""
    if "tests.offline" in sys.modules:
        return False
    return (sys.platform.startswith("linux") and bool(shutil.which("podman"))
            and str(_load().get("EMBED_ONDEMAND", "")).lower() in ("on", "1", "true", "yes"))


def local(url):
    """Whether a server URL is on this machine."""
    return (urlparse(url).hostname or "") in ("127.0.0.1", "localhost", "::1")


def wanted(role, up, url, clock=time.monotonic):
    """Note that embedmem asked for this role's server, and start its
    container in the background if it's down. Returns at once."""
    if role not in ROLE_OF or not local(url):
        return
    with _lock:
        _last[role] = clock()
        start = not up and role not in _starting
        if start:
            _starting.add(role)
    if start:
        threading.Thread(target=_start, args=(role,), name=f"ondemand-{role}", daemon=True).start()
    _ensure_reaper()


def _start(role):
    try:
        results = lifecycle.up(_load(), (role,), keys.token_source()[0], out=lambda *a: None)
        if any(r.role == role and r.state == "started" for r in results):
            with _lock:
                _started.add(role)
    except Exception:
        pass                  # the turn already fell back; the next one asks again
    finally:
        with _lock:
            _starting.discard(role)


def idle_roles(now):
    """The roles this started that nobody has wanted for their idle time."""
    s = _load()
    out = []
    with _lock:
        for role in sorted(_started):
            key, default = ROLE_OF[role]
            try:
                limit = float(s.get(key) or default)
            except ValueError:
                limit = default
            if now - _last.get(role, now) >= limit:
                out.append(role)
        _started.difference_update(out)
    return out


def reap(clock=time.monotonic):
    """Stop what's been idle (one pass; the reaper thread repeats it)."""
    roles = idle_roles(clock())
    if roles:
        lifecycle.down(tuple(roles))
    return roles


def _ensure_reaper():
    global _reaper
    with _lock:
        if _reaper is not None:
            return
        _reaper = threading.Thread(target=_reap_forever, name="ondemand-reaper", daemon=True)
    _reaper.start()


def _reap_forever():
    while True:
        time.sleep(REAP_EVERY)
        try:
            reap()
        except Exception:
            pass
