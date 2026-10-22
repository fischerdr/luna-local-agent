"""On-demand management for embedding and reranker llama-server side processes.

The chat model is always loaded. Embedding and reranker are small servers
on separate ports that only run when knowledge retrieval is needed. They
start on first use and stop after IDLE_TIMEOUT seconds of inactivity,
freeing VRAM for the larger chat model.

Usage:
    import llama_side

    # Start embedding on demand (blocking until ready)
    llama_side.ensure_embed()       # port 8081

    # Start reranker on demand
    llama_side.ensure_rerank()      # port 8082

    # Check if a server is answering
    if llama_side.is_up("embed"):
        ...

    # Force-stop a server (e.g. on exit)
    llama_side.stop_embed()
"""
import os
import subprocess
import threading
import time

import requests

import config
import logbook

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONTAINER_NAME = "luna-llama"
LLAMA_SERVER = "/app/llama-server"   # path inside the container

EMBED_PORT = int(getattr(config, "EMBED_PORT", 8081))
RERANK_PORT = int(getattr(config, "RERANK_PORT", 8082))
EMBED_CTX = int(getattr(config, "EMBED_CTX", 2048))
RERANK_CTX = int(getattr(config, "RERANK_CTX", 2048))
NGL = int(getattr(config, "NGL", 0))       # side servers don't need GPU offload
THREADS = int(getattr(config, "THREADS", 9))

# How long to keep a server running after its last request.
EMBED_IDLE_TIMEOUT = 180      # 3 minutes
RERANK_IDLE_TIMEOUT = 60      # 1 minute

# How long to wait for a freshly-started server to be ready.
START_TIMEOUT = 90

# ---------------------------------------------------------------------------
# Model paths (read from server.env or scan models directory)
# ---------------------------------------------------------------------------

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LLAMA_DIR = os.path.join(ROOT, "llama")
MODEL_DIR = os.path.join(LLAMA_DIR, "models")
SERVER_ENV = os.path.join(LLAMA_DIR, "server.env")


def _read_server_env():
    """Parse server.env for MODEL, EMBED_MODEL, RERANK_MODEL, HOST.

    Handles inline comments and quoted values.
    """
    result = {"HOST": "127.0.0.1", "EMBED_MODEL": "", "RERANK_MODEL": "",
              "EMBED_MATCH": "", "RERANK_MATCH": ""}
    if not os.path.isfile(SERVER_ENV):
        return result
    with open(SERVER_ENV) as f:
        for line in f:
            line = line.strip()
            if "=" not in line or line.startswith("#"):
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            # Strip inline comments (but be careful with values containing #)
            if "#" in val:
                val = val[:val.index("#")]
            val = val.strip().strip('"').strip("'")
            if key in result:
                result[key] = val
    return result


def _find_gguf(pattern):
    """Find a GGUF file in models/ matching *pattern* (case-insensitive)."""
    if not os.path.isdir(MODEL_DIR):
        return None
    import fnmatch
    pattern_lower = pattern.lower()
    for name in os.listdir(MODEL_DIR):
        if not name.lower().endswith(".gguf"):
            continue
        if fnmatch.fnmatch(name.lower(), pattern_lower):
            return os.path.join(MODEL_DIR, name)
    return None


def _resolve_model(env, key, match_key):
    """Return the full path to a model, or None."""
    model = env.get(key, "")
    if model and os.path.isfile(model):
        return model
    match = env.get(match_key, "")
    if match:
        return _find_gguf(match)
    return None


# ---------------------------------------------------------------------------
# Server state (thread-safe)
# ---------------------------------------------------------------------------

_servers = {}           # port -> {"running": bool, "timer": Timer|None}
_lock = threading.Lock()


def _server_state(port):
    with _lock:
        return _servers.setdefault(port, {"running": False, "timer": None})


# ---------------------------------------------------------------------------
# Podman exec helpers
# ---------------------------------------------------------------------------

def _podman_exec(cmd, timeout=START_TIMEOUT):
    """Run a podman exec command, return (returncode, stdout, stderr)."""
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _is_container_running():
    """Check if the llama container is up."""
    r = subprocess.run(
        ["podman", "ps", "-a", "--filter", f"name={CONTAINER_NAME}",
         "--format", "{{.Status}}"],
        capture_output=True, text=True, timeout=5,
    )
    return "Up" in r.stdout


def _port_inet(port):
    """Convert a port number to the 4-digit hex used in /proc/net/tcp."""
    return format(port, "04X")


def _pid_for_port(port):
    """Find the PID inside the container that owns the socket for *port*.

    Uses /proc/net/tcp to find the LISTEN inode for this port, then walks
    /proc/[pid]/fd symlinks to find which process holds it.
    Returns the PID or None.
    """
    port_hex = _port_inet(port)
    try:
        # Step 1: find the inode for this port using awk
        #   $2 = local_address (matches :PORT$), $4 = state (must be 0A = LISTEN)
        #   print $11 = inode
        awk_prog = (
            "$2 ~ /:" + port_hex + "$/ && $4 == \"0A\" {print $10}"
        )
        result = _podman_exec(
            ["podman", "exec", CONTAINER_NAME, "awk", awk_prog,
             "/proc/net/tcp"],
            timeout=5,
        )
        inodes = [i.strip() for i in result.stdout.strip().splitlines()
                  if i.strip() and i.strip() != "0"]
        if not inodes:
            return None

        # Step 2: find which PID holds any of these inodes
        inode_arg = " ".join(f'"{i}"' for i in inodes)
        pid_cmd = (
            "for inode in " + inode_arg + "; do "
            "for p in $(ls /proc | grep -E '^[0-9]+$'); do "
            "if ls -l /proc/$p/fd 2>/dev/null | grep -q \"socket:\\[$inode\\]\"; then "
            "echo $p; exit 0; fi; done; done"
        )
        result2 = _podman_exec(
            ["podman", "exec", CONTAINER_NAME, "bash", "-c", pid_cmd],
            timeout=10,
        )
        pid = result2.stdout.strip()
        if pid:
            return int(pid)
    except Exception:
        pass
    return None


def _wait_for_server(port, timeout=START_TIMEOUT):
    """Block until the server on *port* answers /v1/models."""
    env = _read_server_env()
    host = env.get("HOST", "127.0.0.1")
    url = f"http://{host}:{port}"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            r = requests.get(f"{url}/v1/models", timeout=3)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


# ---------------------------------------------------------------------------
# Start / stop
# ---------------------------------------------------------------------------

def start_server(port, model_path, extra_args, idle_timeout=EMBED_IDLE_TIMEOUT):
    """Start a llama-server inside the container on *port*. Returns True on success."""
    state = _server_state(port)
    if state["running"]:
        # Already running - reset the idle timer
        _reset_timer(port, idle_timeout)
        return True

    if not _is_container_running():
        logbook.warn("embed", f"container {CONTAINER_NAME} not running")
        return False

    # Translate host model path to container path
    # Models are mounted as: llama/models/ -> /models/
    model_path = os.path.join("/models", os.path.basename(model_path))

    # Build the podman exec command
    args = [
        "podman", "exec", "-d", CONTAINER_NAME,
        LLAMA_SERVER,
        "-m", model_path,
        "--host", "0.0.0.0", "--port", str(port),
        "--api-key-file", "/data/.api_key",
        "--cors-origins", "localhost",
    ]
    if extra_args:
        args.extend(extra_args)

    r = _podman_exec(args)
    if r.returncode != 0:
        logbook.warn("embed", f"failed to start server on :{port}: {r.stderr[:200]}")
        return False

    logbook.info("embed", f"side server starting on :{port} ({os.path.basename(model_path)})")
    state["running"] = True

    # Wait for the model to load
    ready = _wait_for_server(port, timeout=START_TIMEOUT)
    if not ready:
        logbook.warn("embed", f"side server on :{port} did not become ready in {START_TIMEOUT}s")
        state["running"] = False
        return False

    logbook.info("embed", f"side server up on :{port} ({os.path.basename(model_path)})")
    _schedule_stop(port, idle_timeout)
    return True


def stop_server(port):
    """Stop the side server on *port* by killing only the process listening
    on that port.  Does NOT affect the chat server on any other port."""
    state = _server_state(port)
    timer = state.get("timer")
    if timer:
        timer.cancel()
        state["timer"] = None

    if not state["running"]:
        return True

    # Find the PID on this port
    pid = _pid_for_port(port)

    if pid is not None and pid != 1:
        # Safe to kill: it's the side server, not PID 1 (the chat server)
        _podman_exec(
            ["podman", "exec", CONTAINER_NAME, "kill", "-TERM", str(pid)],
            timeout=10,
        )
        time.sleep(3)

        # Escalate if still alive
        try:
            if requests.get(f"http://127.0.0.1:{port}/v1/models", timeout=2).status_code == 200:
                _podman_exec(
                    ["podman", "exec", CONTAINER_NAME, "kill", "-KILL", str(pid)],
                    timeout=10,
                )
        except Exception:
            pass

    elif pid == 1:
        # Should never happen for side servers, but be safe
        logbook.error("embed", f"port {port} is PID 1 (chat server) - refusing to stop")
        return False

    else:
        # PID not found - server may already be down
        logbook.info("embed", f"port {port} PID not found, server likely already stopped")

    state["running"] = False
    logbook.info("embed", f"side server stopped on :{port}")
    return True


def is_up(port):
    """Is the server on *port* answering?"""
    state = _server_state(port)
    if not state["running"]:
        return False

    env = _read_server_env()
    host = env.get("HOST", "127.0.0.1")
    try:
        r = requests.get(f"http://{host}:{port}/v1/models", timeout=2)
        if r.status_code == 200:
            # Still responding - reset idle timer
            idle_timeout = EMBED_IDLE_TIMEOUT if port == EMBED_PORT else RERANK_IDLE_TIMEOUT
            _reset_timer(port, idle_timeout)
            return True
    except Exception:
        pass

    # Server died - clean up
    state["running"] = False
    return False


# ---------------------------------------------------------------------------
# Idle management
# ---------------------------------------------------------------------------

def _schedule_stop(port, timeout):
    """Schedule the server to stop after *timeout* seconds of inactivity."""
    state = _server_state(port)
    timer = state.get("timer")
    if timer:
        timer.cancel()

    def _do_stop():
        with _lock:
            if _server_state(port)["running"]:
                stop_server(port)

    t = threading.Timer(timeout, _do_stop)
    t.daemon = True
    state["timer"] = t
    t.start()


def _reset_timer(port, timeout):
    """Reset the idle timer for *port*."""
    state = _server_state(port)
    timer = state.get("timer")
    if timer:
        timer.cancel()
    _schedule_stop(port, timeout)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ensure_embed():
    """Ensure the embedding server is running. Returns True on success."""
    env = _read_server_env()
    model = _resolve_model(env, "EMBED_MODEL", "EMBED_MATCH")
    if not model:
        logbook.warn("embed", "no embedding model found")
        return False

    pooling = getattr(config, "EMBED_POOLING", "last")
    extra = ["--embeddings", "--pooling", pooling,
             "-c", str(EMBED_CTX), "-b", str(EMBED_CTX), "-ub", str(EMBED_CTX)]

    return start_server(EMBED_PORT, model, extra, EMBED_IDLE_TIMEOUT)


def ensure_rerank():
    """Ensure the reranker server is running. Returns True on success."""
    env = _read_server_env()
    model = _resolve_model(env, "RERANK_MODEL", "RERANK_MATCH")
    if not model:
        logbook.warn("embed", "no reranker model found")
        return False

    extra = ["--reranking", "--pooling", "rank",
             "-c", str(RERANK_CTX), "-b", str(RERANK_CTX), "-ub", str(RERANK_CTX)]

    return start_server(RERANK_PORT, model, extra, RERANK_IDLE_TIMEOUT)


def stop_embed():
    """Stop the embedding server."""
    return stop_server(EMBED_PORT)


def stop_rerank():
    """Stop the reranker server."""
    return stop_server(RERANK_PORT)


def shutdown_all():
    """Stop all side servers (call on exit)."""
    stop_rerank()
    stop_embed()
