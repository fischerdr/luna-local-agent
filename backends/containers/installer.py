"""The containers backend in the installer (backend.py's hooks call these).

--backend container / --voice container run the model servers and the
voice in rootless Podman containers on this machine: python -m
backends.containers does the work, in Luna's own Python. --remote ADDR
points her at a server running deploy/ instead. `inst` is the installer
(installer/install.py), for its flags and helpers.
"""
import ipaddress
import os
import re
import shutil
import socket
import subprocess
import sys
import urllib.error
import urllib.request

from backends.containers import HERE, ROOT

SIDE_EXTRAS = ("embeddings", "reranker")
REMOTE_PORTS = (("chat", 8080, "/v1/models"), ("embeddings", 8081, "/v1/models"),
                ("reranker", 8082, "/v1/models"), ("voice", 8899, "/health"))
LOCAL_ENV = os.path.join(HERE, "services.local.env")


def url_host(addr):
    """An address as it goes in a URL: IPv6 in brackets."""
    return f"[{addr}]" if ":" in addr and not addr.startswith("[") else addr


def check_address(addr):
    """None if addr can be a server on the network, else why not."""
    bare = addr.strip("[]")
    try:
        ip = ipaddress.ip_address(bare)
    except ValueError:
        if not re.fullmatch(r"(?=.{1,253}$)([A-Za-z0-9]([A-Za-z0-9-]{0,61}[A-Za-z0-9])?)(\.[A-Za-z0-9]([A-Za-z0-9-]{0,61}[A-Za-z0-9])?)*", bare):
            return f"{addr} isn't an IP address or a host name"
        if bare.lower() == "localhost":
            return "that's this machine - for servers here, use --backend container (or llama)"
        return None
    if ip.is_loopback or ip.is_unspecified:
        return "that's this machine - for servers here, use --backend container (or llama)"
    return None


def validate(inst, args, system=None, which=shutil.which, exists=os.path.exists):
    """Its flags against the others, before anything runs: (errors, notes).
    Fills in --backend remote for --remote, the server's voice for it, and
    leaves out the extras the containers or the server bring themselves."""
    errors, notes = [], []
    system = system or inst.SYSTEM
    if args.check:
        if args.remote:
            notes.append("--check only runs the health check - ignoring --remote")
        return errors, notes

    if args.remote:
        if args.backend not in (None, "remote"):
            errors.append(f"--remote goes with --backend remote, not --backend {args.backend}")
        else:
            args.backend = "remote"
            why = check_address(args.remote)
            if why:
                errors.append(f"--remote {args.remote}: {why}")
    elif args.backend == "remote" and args.yes:
        errors.append("--backend remote needs the server's address: --remote <address>")

    containers = args.backend == "container" or args.voice == "container"
    if containers and system != "Linux":
        errors.append("containers here are Linux only - on this machine use LM Studio, "
                      "or put them on a Linux GPU server and use --remote <its address>")
    if args.gpu == "mps" and containers:
        errors.append("--gpu mps is Apple's GPU - containers can't use it")
    elif containers and args.gpu == "cuda" and not which("nvidia-smi"):
        notes.append("--gpu cuda, but there's no NVIDIA GPU here - the containers use the cpu image")
    elif containers and args.gpu == "rocm" and not exists("/dev/kfd"):
        notes.append("--gpu rocm, but there's no /dev/kfd here - the containers use the cpu image")

    if args.extras is not None and args.backend in ("container", "remote"):
        names = list(inst.EXTRAS) if args.extras == "all" else [e.strip() for e in args.extras.split(",")]
        dropped = [e for e in names if e in SIDE_EXTRAS]
        if dropped:
            where = "the containers download them by themselves" if args.backend == "container" \
                else "the server holds its own models"
            notes.append(f"leaving out {', '.join(dropped)} - {where}")
        args.extras = ",".join(e for e in names if e and e not in SIDE_EXTRAS)

    if args.backend == "remote" and args.remote and args.voice is None and not errors:
        args.voice = f"url:http://{url_host(args.remote)}:8899"
    return errors, notes


def services(inst, state, *args):
    """python -m backends.containers, in Luna's Python (it needs requests). True on success."""
    py = state.get("venv_python") or sys.executable
    return inst.run([py, "-m", "backends.containers", *args, *(["--yes"] if inst.ARGS.yes else [])], cwd=ROOT)


def container_gpu(inst):
    """The --gpu flag as python -m backends.containers setup takes it."""
    gpu = inst.ARGS.gpu
    if gpu == "cuda":
        return "auto" if shutil.which("nvidia-smi") else "cpu"
    if gpu == "rocm":
        return "rocm" if os.path.exists("/dev/kfd") else "cpu"
    return {"cpu": "cpu"}.get(gpu, "auto")


def containers(inst, state):
    """--backend container: python -m backends.containers setup, after the model step."""
    inst.step("Her brain, in containers")
    if not shutil.which("podman"):
        inst.warn("Podman isn't installed - on Fedora: sudo dnf install podman; Ubuntu/Debian: sudo apt install podman")
        print("  Or run the installer again with --backend lmstudio, or --backend llama to build llama.cpp here.")
        inst.note("model server", False, "containers: Podman isn't installed")
        return
    good = services(inst, state, "setup", "--gpu", container_gpu(inst))
    inst.note("model server", good, "in containers - python -m backends.containers up starts them"
              if good else "container setup stopped - the lines above say why")


def voice_container(inst, state):
    """--voice container: python -m backends.containers setup tts. True if it worked."""
    if not shutil.which("podman"):
        inst.warn("Podman isn't installed - the voice container needs it "
                  "(Fedora: sudo dnf install podman; Ubuntu/Debian: sudo apt install podman)")
        inst.note("voice", False, "container: Podman isn't installed")
        return False
    good = services(inst, state, "setup", "tts")
    if good:
        # setup tts wrote install.json's voice; keep it when this state is saved
        state["voice"] = inst.load_state().get("voice") or state.get("voice", {})
    inst.note("voice", good, "in a container (luna-tts) on :8899" if good else "voice container setup stopped")
    return good


def gpu_reserve(state):
    """GB the containers keep on the GPU beside her model: her voice's
    (about 1 GB), when it runs on an NVIDIA card with a CDI spec. Chat and
    voice come first; the reranker and embedder take what's left, or the
    CPU. Nothing for any other backend."""
    if state.get("backend") != "container":
        return None
    voice_gpu = (state.get("voice", {}).get("container") and shutil.which("nvidia-smi")
                 and any(os.path.exists(p) for p in ("/etc/cdi/nvidia.yaml",
                                                     os.path.expanduser("~/.config/cdi/nvidia.yaml"))))
    return 1.0 if voice_gpu else None


def set_env_line(path, key, value):
    """Set KEY="value" in a .local.env file, adding or replacing its line."""
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except OSError:
        text = "# Written by the installer for this machine (not committed). Edit freely.\n"
    line = f'{key}="{value}"'
    pattern = re.compile(rf"^{re.escape(key)}=.*$", re.M)
    text = pattern.sub(line, text, count=1) if pattern.search(text) else text.rstrip("\n") + "\n" + line + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def container_model(inst, state, pick):
    """The picked model for the containers: downloaded by -hf into their
    volume on the first start, so nothing lands in llama/models."""
    # the quant is the file's last part: Qwen_Qwen3.5-9B-Q6_K.gguf, or the
    # uncensored ones' Huihui-Qwen3.5-9B-abliterated.Q6_K.gguf
    m = re.search(r"[-.]((?:I?Q\d\w*)|F16|BF16)\.gguf$", pick["file"], re.I)
    hf = f"{pick['repo']}:{m.group(1)}" if m else pick["repo"]
    set_env_line(LOCAL_ENV, "MODEL_HF", hf)
    local = os.path.join(ROOT, "llama", "server.local.env")
    set_env_line(local, "CTX", str(pick["ctx"]))
    # the small picks leave vision out, so the model and its context fit
    set_env_line(local, "MMPROJ", "" if pick["mmproj"] else "off")
    state["model"] = pick["key"]
    inst.ok(f"MODEL_HF={hf} - in backends/containers/services.local.env; it downloads on the first start")
    inst.note("model", True, f"{pick['label']}, for the containers")


def client_ip(host):
    """This machine's address as the server sees it (for the firewall rule)."""
    try:
        family = socket.AF_INET6 if ":" in host.strip("[]") else socket.AF_INET
        with socket.socket(family, socket.SOCK_DGRAM) as s:
            s.connect((host.strip("[]"), 9))
            return s.getsockname()[0]
    except OSError:
        return "this machine's address"


def remote_probe(host, key, get=None, ports=REMOTE_PORTS):
    """Each service on the server: [(name, ok, line)]."""
    lines = []
    for name, port, path in ports:
        url = f"http://{url_host(host)}:{port}{path}"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"} if key else {})
        try:
            with (get or urllib.request.urlopen)(req, timeout=3) as r:
                code = r.status
        except urllib.error.HTTPError as e:
            code = e.code
        except Exception:
            code = None
        if code == 200:
            lines.append((name, True, f"{name}: {url}"))
        elif code == 401:
            lines.append((name, False, f"{name}: the key doesn't match the server's (llama/.api_key)"))
        elif code is None:
            lines.append((name, False, f"{name}: no answer at {url} - check that the server is up, that its "
                                       f"LUNA_BIND is its own address, and that its firewall lets this "
                                       f"machine in ({client_ip(host)})"))
        else:
            lines.append((name, False, f"{name}: {url} answered {code}"))
    return lines


def write_key(key):
    """llama/.api_key, readable only by you."""
    path = os.path.join(ROOT, "llama", ".api_key")
    old = os.umask(0o077)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(key.strip())
    finally:
        os.umask(old)
    os.chmod(path, 0o600)
    return path


def read_key():
    try:
        with open(os.path.join(ROOT, "llama", ".api_key"), encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return ""


def remote(inst, state):
    """--backend remote: her URLs on the server, its key here, then a probe."""
    inst.step("Her brain, on another machine")
    host = inst.ARGS.remote
    while not host:
        try:
            host = input("  The server's address on your network: ").strip()
        except EOFError:
            host = ""
        if not host:
            inst.warn("no address - run again with --remote <address>")
            inst.note("model server", False, "remote: no address")
            return
        why = check_address(host)
        if why:
            inst.warn(why)
            host = None
    base = f"http://{url_host(host)}"
    inst.set_config("llm", "backend", "llama")
    inst.set_config("llm", "llama_url", f"{base}:8080/v1/chat/completions")
    inst.set_config("llm", "embed_url", f"{base}:8081/v1/embeddings")
    inst.set_config("llm", "rerank_url", f"{base}:8082/v1/rerank")
    # start_llama() doesn't ask where the URL points: with a llama.cpp
    # build here it would start a useless local server whenever the
    # remote one is down.
    inst.set_config("llm", "autostart", False)
    state["remote"] = host
    inst.ok(f"config.json: her model, embeddings and reranker at {base}")

    have = read_key()
    if inst.ARGS.yes:
        if not have:
            inst.warn("no key yet - copy the server's backends/containers/deploy/api_key to llama/.api_key")
    else:
        try:
            given = input("  The server's api_key (paste it, or a path to the file; Enter keeps "
                          f"{'the one here' if have else 'none'}): ").strip()
        except EOFError:
            given = ""
        if given and os.path.isfile(os.path.expanduser(given)):
            with open(os.path.expanduser(given), encoding="utf-8") as f:
                given = f.read().strip()
        if given and given != have and (not have or inst.ask("Replace the key in llama/.api_key?", default=False)):
            write_key(given)
            have = given
            inst.ok("llama/.api_key written (readable only by you)")

    results = remote_probe(host, have)
    for _name, good, line in results:
        (inst.ok if good else inst.fail)(line)
    up = all(good for _n, good, _l in results[:1])
    inst.note("model server", up, f"on {host}" if up else f"on {host} - not answering yet (see above)")


def health(inst, state):
    """The health check's lines: each container, or the server's services."""
    if state.get("backend") == "container" or state.get("voice", {}).get("container"):
        py = state.get("venv_python") or sys.executable
        out = subprocess.run([py, "-m", "backends.containers", "status"], cwd=ROOT, capture_output=True, text=True)
        for line in (out.stdout or out.stderr).strip().splitlines():
            print("  " + line.strip().replace("✓", f"{inst.G}✓{inst.X}").replace("·", f"{inst.Y}·{inst.X}"))
    if state.get("backend") == "remote" and state.get("remote"):
        for _name, good, line in remote_probe(state["remote"], read_key())[:3]:
            (inst.ok if good else inst.warn)(line)
