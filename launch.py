"""Start Luna - and the servers she needs first, if they aren't up.

    ./start.sh   /   luna   /   luna.cmd   (they all end up here)

Any arguments are passed on to main.py. Before her:

- the model server: llama/start.sh, when llama.cpp is built here and
  neither it nor LM Studio is answering. It picks server.env, or
  server.mac.env on a Mac, by itself. Off with "llm": {"autostart":
  false}, or llm.backend "lmstudio".
- the voice server, if the installer set one up.

Each is started only if it isn't already answering, and stopped again
when she exits - one that was already running before is left alone.

Your settings are config.json, which is yours and never in git: the
first start makes it from config.example.json, the shipped defaults.

First it checks that every package in installer/requirements.txt is
installed - after a pull, a new one often isn't - and offers to install
what's missing, instead of her stopping with a traceback.

    ./start.sh --check    check packages and servers, say what's missing, exit
    ./start.sh --update   install missing packages, then check (update.sh runs this)
"""
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
WIN = os.name == "nt"


def up(url):
    try:
        with urllib.request.urlopen(url, timeout=1.5) as r:
            return r.status < 500
    except Exception:
        return False


def state():
    try:
        with open(os.path.join(ROOT, "install.json"), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def log_dir():
    base = os.environ.get("LOCALAPPDATA") if WIN else os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    path = os.path.join(base or os.path.expanduser("~"), "ai-voice")
    os.makedirs(path, exist_ok=True)
    return path


LLAMA_DIR = os.path.join(ROOT, "llama")


CONFIG = os.path.join(ROOT, "config.json")
EXAMPLE = os.path.join(ROOT, "config.example.json")


def seed_config():
    """config.json is yours and stays out of git; a fresh clone (or an
    update that took the old tracked copy away) gets the shipped defaults.
    Never overwrites one that's there. True if it made one."""
    if os.path.exists(CONFIG) or not os.path.exists(EXAMPLE):
        return False
    import shutil

    shutil.copyfile(EXAMPLE, CONFIG)
    return True


def settings():
    try:
        with open(CONFIG, encoding="utf-8") as f:
            found = json.load(f)
        return found if isinstance(found, dict) else {}
    except (OSError, ValueError):
        return {}


def llm_settings():
    return settings().get("llm", {}) or {}


def tts_url():
    """Where her voice is: config.json's tts.url (a kokoro-reader on another
    machine counts), else what the installer set up, else this machine."""
    url = (settings().get("tts") or {}).get("url") or state().get("voice", {}).get("url")
    return str(url or "http://127.0.0.1:8899").rstrip("/")


def _here(url):
    from urllib.parse import urlparse

    return (urlparse(url).hostname or "") in ("127.0.0.1", "localhost", "::1")


def model_up():
    """Is a model server answering - at the addresses in config.json,
    so a server on another machine counts too."""
    llm = llm_settings()
    llama = str(llm.get("llama_url") or "http://127.0.0.1:8080/v1/chat/completions").split("/v1/")[0]
    studio = str(llm.get("lmstudio_url") or "http://localhost:1234/v1/chat/completions").split("/v1/")[0]
    return up(llama + "/health") or up(studio + "/v1/models")


def start_llama(s):
    """llama/start.sh in the background (its embedding and reranker
    servers too), in its own process group so one signal stops all of
    them. Returns the process, or None if it wasn't ours to start."""
    llm = llm_settings()
    backend = str(llm.get("backend", "auto")).lower()
    if WIN or llm.get("autostart", True) is False or backend.startswith("lm") or s.get("backend") == "lmstudio":
        return None
    script = os.path.join(LLAMA_DIR, "start.sh")
    if model_up() or not os.path.exists(script) or not os.access(os.path.join(LLAMA_DIR, "bin", "llama-server"), os.X_OK):
        return None

    log_path = os.path.join(LLAMA_DIR, "server.log")
    log = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(["bash", script], cwd=LLAMA_DIR, stdout=log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True)
    print("Starting her model (llama.cpp)", end="", flush=True)
    for _ in range(600):           # a big model from a slow disk can take a few minutes
        if proc.poll() is not None:
            print(f" it stopped - last lines of llama/server.log:")
            try:
                with open(log_path, encoding="utf-8", errors="replace") as f:
                    print("  " + "\n  ".join(f.read().splitlines()[-8:]))
            except OSError:
                pass
            return None
        if up("http://127.0.0.1:8080/health"):   # 503 while loading, 200 when ready
            print(" ready.")
            return proc
        time.sleep(0.5)
        if _ % 4 == 0:
            print(".", end="", flush=True)
    print(" still loading - see llama/server.log.")
    return proc


def stop_group(proc):
    """Stop a process and everything it started (llama/start.sh's side servers)."""
    if not proc or proc.poll() is not None:
        return
    try:
        if WIN:
            proc.terminate()
        else:
            import signal
            os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(8)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def start_voice(v):
    """kokoro-reader in the background. Returns the process, or None."""
    port = int(v.get("port", 8899))
    if up(f"http://127.0.0.1:{port}/health"):
        return None
    if not (os.path.exists(v.get("python", "")) and os.path.exists(v.get("server", ""))):
        print("Her voice server isn't where the installer left it - run the installer again.", file=sys.stderr)
        return None
    # Always passed, "cpu" included: the server's own default is "auto",
    # which picks the GPU - and on Apple silicon that's the device the
    # installer deliberately steered away from.
    env = dict(os.environ, KOKORO_HOST="127.0.0.1", KOKORO_PORT=str(port),
               KOKORO_DEVICE=v.get("device", "cpu"))
    log = open(os.path.join(log_dir(), "kokoro.log"), "a", encoding="utf-8")
    flags = subprocess.CREATE_NO_WINDOW if WIN else 0
    proc = subprocess.Popen([v["python"], v["server"]], cwd=os.path.dirname(v["server"]), env=env,
                            stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            creationflags=flags)
    print("Starting her voice...", end="", flush=True)
    for _ in range(40):            # the model loads on the first start - give it a moment
        if proc.poll() is not None:
            print(f" it stopped - see {log.name}")
            return None
        if up(f"http://127.0.0.1:{port}/health"):
            print(" ready.")
            return proc
        time.sleep(0.5)
        print(".", end="", flush=True)
    print(" still loading (first start downloads the model) - she'll find it when it's up.")
    return proc


# ---------------------------------------------------------------------------
# Container voice mode (Podman)
# ---------------------------------------------------------------------------

VOICE_IMAGE = "luna-tts"
VOICE_CONTAINER = "luna-tts"


def _podman_run(args, check=True):
    """Run a podman command."""
    try:
        r = subprocess.run(["podman"] + args, capture_output=True, text=True, timeout=30)
        if check and r.returncode != 0:
            print(f"  podman {' '.join(args)}: {r.stderr[:200]}", file=sys.stderr)
        return r
    except FileNotFoundError:
        return None
    except subprocess.TimeoutExpired:
        return None


def _container_voice_proc(mode):
    """Start the voice server via Podman container. Returns None (container is separate).
    
    Checks if the container is running, starts it if stopped, or builds + starts
    if it doesn't exist yet.
    """
    port = 8899
    if up(f"http://127.0.0.1:{port}/health"):
        return None
    
    image_tag = f"{VOICE_IMAGE}:{mode}"
    name = VOICE_CONTAINER
    
    # Check existing container status
    r = _podman_run(["ps", "-a", "--filter", f"name={name}", "--format", "{{.Status}}"])
    status = r.stdout.strip() if r else ""
    
    if not status:
        # Container doesn't exist — try to build it from Dockerfile
        print("  voice container not found. Building...")
        tmpdir = os.path.join(ROOT, "tts", ".build-tmp")
        os.makedirs(tmpdir, exist_ok=True)
        try:
            # Clone kokoro-reader if needed
            kokoro_dir = os.environ.get("LUNA_KOKORO_DIR",
                                        os.path.join(os.path.dirname(ROOT), "kokoro-reader"))
            server_dir = os.path.join(kokoro_dir, "server")
            has_server = os.path.isfile(os.path.join(server_dir, "kokoro_server.py"))
            if not has_server:
                print("  cloning kokoro-reader...")
                r = subprocess.run(["git", "clone", "--depth", "1",
                                    os.environ.get("LUNA_KOKORO_REPO",
                                                   "https://github.com/fswolf/kokoro-reader.git"),
                                    kokoro_dir],
                                   capture_output=True, text=True)
                if r.returncode != 0:
                    print("  voice container: clone failed", file=sys.stderr)
                    return None
            
            # Copy server files into build context
            for fname in ["kokoro_server.py", "blends.json", "start_server.sh"]:
                src = os.path.join(server_dir, fname)
                if os.path.isfile(src):
                    shutil.copy2(src, tmpdir)
            
            # Write Dockerfile to build context
            dockerfile_name = "Dockerfile.cuda" if mode == "container_cuda" else "Dockerfile"
            df_src = os.path.join(ROOT, "tts", dockerfile_name)
            if os.path.isfile(df_src):
                shutil.copy2(df_src, os.path.join(tmpdir, "Dockerfile"))
            else:
                # Inline fallback
                if mode == "container_cuda":
                    content = (
                        "FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04\n"
                        "RUN apt-get update && apt-get install -y --no-install-recommends "
                        "git espeak-ng python3-pip && rm -rf /var/lib/apt/lists/* && rm -f /usr/lib/python3.12/EXTERNALLY-MANAGED 2>/dev/null; true\n"
                        "RUN git clone --depth 1 https://github.com/fswolf/kokoro-reader.git /opt/kokoro-reader\n"
                        "RUN pip install --no-cache-dir --break-system-packages torch "
                        "ctranslate2 kokoro>=0.9.4 numpy>=1.24 soundfile\n"
                        "WORKDIR /opt/kokoro-reader/server\n"
                        "ENV KOKORO_HOST=0.0.0.0 KOKORO_PORT=8899 KOKORO_LANG=a "
                        "KOKORO_DEVICE=cuda KOKORO_VOICE=am_adam\n"
                        "EXPOSE 8899\n"
                        "CMD [\"python3\", \"kokoro_server.py\"]\n"
                    )
                else:
                    content = (
                        "FROM python:3.12-slim\n"
                        "RUN apt-get update && apt-get install -y --no-install-recommends "
                        "git espeak-ng && rm -rf /var/lib/apt/lists/* && rm -f /usr/lib/python3.12/EXTERNALLY-MANAGED 2>/dev/null; true\n"
                        "RUN git clone --depth 1 https://github.com/fswolf/kokoro-reader.git /opt/kokoro-reader\n"
                        "RUN pip install --no-cache-dir --break-system-packages "
                        "\"torch[cpu]\" ctranslate2 kokoro>=0.9.4 numpy>=1.24 soundfile\n"
                        "WORKDIR /opt/kokoro-reader/server\n"
                        "ENV KOKORO_HOST=0.0.0.0 KOKORO_PORT=8899 KOKORO_LANG=a "
                        "KOKORO_DEVICE= KOKORO_VOICE=am_adam\n"
                        "EXPOSE 8899\n"
                        "CMD [\"python3\", \"kokoro_server.py\"]\n"
                    )
                with open(os.path.join(tmpdir, "Dockerfile"), "w") as f:
                    f.write(content)
            
            print(f"  building {image_tag}")
            r = subprocess.run(["podman", "build", "-t", image_tag, tmpdir],
                               capture_output=True, text=True)
            if r.returncode != 0:
                print(f"  voice container build failed: {r.stderr[:300]}", file=sys.stderr)
                return None
            
            print("  container image built.")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
    
    elif "Exited" in status or "exited" in status:
        # Container exists but stopped — restart it
        _podman_run(["start", name])
        print("  container restarted.")
    elif "Up" in status:
        # Already running
        return None
    
    # Start the container
    cmd = ["run", "-d", "--name", name, "--network", "host"]
    if mode == "container_cuda":
        cmd.extend(["--gpus", "all"])
    cmd.append(image_tag)
    
    r = subprocess.run(["podman"] + cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  failed to start voice container: {r.stderr[:200]}", file=sys.stderr)
        return None
    
    print("  voice container started.")
    return None  # Container is managed externally by podman


def stop_voice_container():
    """Stop the voice container (called on app exit)."""
    _podman_run(["stop", VOICE_CONTAINER], check=False)


# ---------------------------------------------------------------------------
# Packages: what installer/requirements.txt asks for, against what's installed
# ---------------------------------------------------------------------------
REQUIREMENTS = os.path.join(ROOT, "installer", "requirements.txt")


def _marker_ok(marker):
    """The one kind of environment marker requirements.txt uses:
    sys_platform == / != "linux". Anything else counts as wanted."""
    m = re.fullmatch(r"\s*sys_platform\s*(==|!=)\s*[\"']([^\"']+)[\"']\s*", marker)
    return True if not m else (sys.platform == m.group(2)) == (m.group(1) == "==")


def wanted_packages():
    names = []
    try:
        with open(REQUIREMENTS, encoding="utf-8") as f:
            lines = f.read().splitlines()
    except OSError:
        return names
    for line in lines:
        spec, _, marker = line.split("#")[0].partition(";")
        m = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9_.\-]*)", spec)
        if m and (not marker.strip() or _marker_ok(marker)):
            names.append(m.group(1))
    return names


def missing_packages(py):
    """The wanted packages Luna's Python doesn't have (pip names), or
    None if that Python couldn't be asked."""
    names = wanted_packages()
    probe = ("import json,sys\nfrom importlib import metadata\nout=[]\n"
             "for n in sys.argv[1:]:\n    try: metadata.version(n)\n"
             "    except metadata.PackageNotFoundError: out.append(n)\nprint(json.dumps(out))")
    try:
        r = subprocess.run([py, "-c", probe, *names], capture_output=True, text=True, timeout=30)
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        return None


def install_packages(py):
    """pip install -r installer/requirements.txt into Luna's Python."""
    print("Installing what's missing...", flush=True)
    return subprocess.call([py, "-m", "pip", "install", "-q", "--disable-pip-version-check",
                            "-r", REQUIREMENTS]) == 0


def ensure_packages(py, ask=True):
    """Say plainly what's missing and offer to install it. True when
    nothing is missing (any more)."""
    missing = missing_packages(py)
    seen = os.path.join(log_dir(), "packages-missing.txt")
    if not missing:
        if os.path.exists(seen):
            os.remove(seen)
        return True
    key = ",".join(sorted(missing))
    try:
        with open(seen, encoding="utf-8") as f:
            asked_before = f.read().strip() == key
    except OSError:
        asked_before = False
    if ask and asked_before:
        # Already offered (declined, or it wouldn't install): one line, not a question every start.
        print(f"Still missing: {', '.join(missing)} - ./start.sh --check says more.")
        return False
    print(f"Luna needs Python packages that aren't installed (new in an update?): {', '.join(missing)}")
    if ask and not (sys.stdin.isatty() and input("Install them now? [Y/n] ").strip().lower() in ("", "y", "yes")):
        print(f"  To install them later: ./update.sh   (or: {py} -m pip install -r installer/requirements.txt)")
    elif install_packages(py) and not missing_packages(py):
        print("Installed.")
        return ensure_packages(py, ask)          # clears the note
    else:
        print("Some didn't install - the lines above say why. The installer can repair it: ./installer/install.sh")
    with open(seen, "w", encoding="utf-8") as f:
        f.write(key)
    return False


# ---------------------------------------------------------------------------
# --check: everything she needs, in plain words, with the fix for each
# ---------------------------------------------------------------------------
def check(py):
    good, lines = True, []

    def line(ok, text, fix=""):
        nonlocal good
        good = good and ok
        lines.append(("  ✓ " if ok else "  ✗ ") + text + ("" if ok or not fix else f"\n      → {fix}"))

    missing = missing_packages(py)
    if missing is None:
        line(False, f"Luna's Python ({py}) didn't answer", "run the installer: ./installer/install.sh")
    else:
        line(not missing, "Python packages" + (f": missing {', '.join(missing)}" if missing else ""),
             "./update.sh  (or: ./start.sh --update)")

    if model_up():
        line(True, "a model server is answering")
    elif os.access(os.path.join(LLAMA_DIR, "bin", "llama-server"), os.X_OK):
        line(True, "no model server running yet - ./start.sh starts llama.cpp for her")
    else:
        line(False, "no model server answering (llama.cpp :8080 or LM Studio :1234)",
             "open LM Studio's Local Server, or build llama.cpp: bash llama/install.sh")

    v = state().get("voice", {})
    url = tts_url()
    if v.get("mode") == "managed" and _here(url):
        ready = os.path.exists(v.get("python", "")) and os.path.exists(v.get("server", ""))
        line(ready, "her voice server is installed - ./start.sh starts it" if ready else
             "her voice server isn't where the installer left it", "run the installer again: ./installer/install.sh")
    elif v.get("mode") in ("container", "container_cuda") and _here(url):
        # Container is started by launch.py; just check if it's answering
        image = v.get("image", f"{VOICE_IMAGE}:{v.get('mode')}")
        health = up(url + "/health")
        line(health,
             f"voice container ({image}) is answering" if health
             else f"voice container ({image}) isn't answering yet",
             "run the installer again: ./installer/install.sh")
    elif v.get("mode") == "none":
        line(True, "no voice server set up - she'll be text-only")
    else:
        line(up(url + "/health"), f"voice server at {url}",
             "start kokoro-reader there, or set tts.url in config.json" if not _here(url)
             else "start kokoro-reader, or run the installer to set one up")

    print("Luna's check:\n" + "\n".join(lines))
    print("All good - ./start.sh" if good else "Fix the ✗ lines above, then ./start.sh")
    return good


def python():
    """The venv's Python: the one the installer made, else whichever runs this."""
    s = state()
    py = s.get("venv_python")
    if py and os.path.exists(py):
        return py
    for cand in (os.environ.get("VIRTUAL_ENV"), os.path.expanduser("~/ai-voice-venv"),
                 os.path.join(ROOT, "ai-voice-venv"), os.path.join(ROOT, "venv"), os.path.join(ROOT, ".venv")):
        if cand:
            p = os.path.join(cand, "Scripts", "python.exe") if WIN else os.path.join(cand, "bin", "python")
            if os.path.exists(p):
                return p
    return sys.executable


def main():
    args = sys.argv[1:]
    if seed_config():
        print("Made config.json from config.example.json - your settings live there, outside git.")
    py = python()
    if "--check" in args:
        sys.exit(0 if check(py) else 1)
    if "--update" in args:
        ensure_packages(py, ask=False)
        print()
        sys.exit(0 if check(py) else 1)

    ensure_packages(py)
    s = state()
    llama = start_llama(s)
    voice = None
    v = s.get("voice", {})
    url = tts_url()
    if v.get("mode") == "managed" and _here(url):   # a voice on another machine needs nothing started here
        voice = start_voice(v)
    elif v.get("mode") in ("container", "container_cuda") and _here(url):
        _container_voice_proc(v["mode"])

    # Register container stop on exit
    if v.get("mode") in ("container", "container_cuda"):
        import atexit
        atexit.register(stop_voice_container)

    warnings = []
    if not model_up():
        warnings.append("No model server - open LM Studio's Local Server, or run llama/start.sh.")
    if not (v.get("mode") in ("managed", "container", "container_cuda") and _here(url)) \
       and v.get("mode") != "none" and not up(url + "/health"):
        warnings.append(f"No voice server at {url} - she'll start text-only.")
    if warnings:
        print("Starting anyway:\n  " + "\n  ".join(warnings), file=sys.stderr)

    try:
        code = subprocess.call([py, os.path.join(ROOT, "main.py"), *args], cwd=ROOT)
    except KeyboardInterrupt:
        code = 130
    finally:
        if voice and voice.poll() is None:
            voice.terminate()
            try:
                voice.wait(5)
            except subprocess.TimeoutExpired:
                voice.kill()
        stop_group(llama)
    sys.exit(code)


if __name__ == "__main__":
    main()
