"""Bringing the containers up and down, and saying how they are.

ensure() is the heart of it, per role: settle the model, write the
container's env-file, work out its podman run - then keep the container
that's there if its config hash matches, or make it anew if not. It
never takes over a port something else holds (a host llama/start.sh,
LM Studio), and never replaces a luna-* container that isn't ours
(one with no luna.config-hash label): setup asks about those.
"""
import grp
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field

import requests

from services import CONTAINER, LLAMA_ROLES, MIN_PODMAN, ROLES, ROOT, VOLUME, images, keys, run, settings


@dataclass
class Context:
    """What every role's podman run shares: the machine and the secrets."""

    backend: str = "cpu"
    cdi_dir: str = None
    selinux: bool = False
    render: bool = False
    key_fp: str = ""
    token_fp: str = ""
    image_ids: dict = field(default_factory=dict)


@dataclass
class Result:
    """What ensure() did with a role, and a line about it."""

    role: str
    state: str          # kept, started, skipped, failed
    note: str = ""


def podman(*args, check=False, **kw):
    """podman, its output captured as text."""
    return subprocess.run(["podman", *args], capture_output=True, text=True, check=check, **kw)


def podman_problem():
    """Why Podman won't do here, in words with the fix - or None."""
    if not shutil.which("podman"):
        return "Podman isn't installed - on Fedora: sudo dnf install podman; on Ubuntu/Debian: sudo apt install podman"
    out = podman("version", "--format", "{{.Client.Version}}").stdout.strip()
    try:
        have = tuple(int(x) for x in re.findall(r"\d+", out)[:3])
    except ValueError:
        have = ()
    if have < MIN_PODMAN:
        need = ".".join(map(str, MIN_PODMAN))
        return f"Podman {out or '?'} is too old - {need} or newer is needed (secret create --replace)"
    if podman("info", "--format", "{{.Host.Security.Rootless}}").stdout.strip() != "true":
        return "Podman isn't running rootless - run python -m services as yourself, not root"
    return None


def selinux():
    """Whether containers here are confined by SELinux labels."""
    return podman("info", "--format", "{{.Host.Security.SELinuxEnabled}}").stdout.strip() == "true"


def render_group():
    """Whether this machine has a render group (GPU access on newer distros)."""
    try:
        grp.getgrnam("render")
        return True
    except KeyError:
        return False


def context(s, token):
    """The Context for this machine and these settings."""
    backend, note = images.image_for_host(s["LLAMA_BACKEND"])
    spec = images.cdi_spec()
    ctx = Context(backend=backend, selinux=selinux(), render=render_group(),
                  cdi_dir=os.path.dirname(spec) if spec == images.CDI_USER else None,
                  key_fp=keys.fingerprint(keys.read_key()), token_fp=keys.fingerprint(token))
    return ctx, note


# --- what's there -----------------------------------------------------------
def containers(ps=None):
    """Every luna-* container, as {name: {"state", "labels"}}."""
    if ps is None:
        out = podman("ps", "-a", "--filter", "name=^luna-", "--format", "json").stdout
        ps = json.loads(out or "[]")
    found = {}
    for c in ps:
        names = c.get("Names") or []
        name = names[0] if isinstance(names, list) and names else str(names)
        if name in CONTAINER.values() or (name.startswith("luna-") and not name.startswith("luna-test-")):
            found[name] = {"state": c.get("State", ""), "labels": c.get("Labels") or {}}
    return found


def unlabelled(found=None):
    """luna-* containers that aren't ours: made by something else (an
    older branch), so never replaced without asking."""
    found = containers() if found is None else found
    return sorted(n for n, c in found.items() if "luna.config-hash" not in c["labels"])


def port_holder(port, ss=None):
    """What's listening on a TCP port, in words - or None when it's free."""
    if ss is None:
        if shutil.which("ss"):
            ss = subprocess.run(["ss", "-ltnpH", f"sport = :{port}"], capture_output=True, text=True).stdout
        else:
            import socket

            with socket.socket() as sock:
                sock.settimeout(0.5)
                return "something" if sock.connect_ex(("127.0.0.1", port)) == 0 else None
    lines = [ln for ln in ss.splitlines() if ln.strip()]
    if not lines:
        return None
    m = re.search(r'users:\(\("([^"]+)",pid=(\d+)', lines[0])
    return f"{m.group(1)} (pid {m.group(2)})" if m else "another program"


def env_dir():
    """Where the env-files go: your runtime dir, never the repo. They hold
    settings, not secrets."""
    base = os.environ.get("XDG_RUNTIME_DIR") or tempfile.gettempdir()
    path = os.path.join(base, f"luna-services-{os.getuid()}")
    os.makedirs(path, mode=0o700, exist_ok=True)
    return path


def wait_ready(url, timeout, get=requests.get, clock=time.monotonic, pause=time.sleep, alive=None):
    """Poll url/health until it says ok (200; 503 means still loading).
    True when ready; False when the time ran out, or alive() says the
    container has stopped (no use waiting for it)."""
    end = clock() + timeout
    while True:
        try:
            if get(url.rstrip("/") + "/health", timeout=2).status_code == 200:
                return True
        except requests.RequestException:
            pass
        if clock() >= end or (alive and not alive()):
            return False
        pause(1)


def running(role):
    """A callable: is the role's container still running?"""
    def alive():
        state = podman("inspect", "-f", "{{.State.Running}}", CONTAINER[role]).stdout.strip()
        return state == "true"
    return alive


# What a container's log says when it can't start, and the fix in words.
_TROUBLE = (
    (re.compile(r"GET failed \((?:401|403)\).*?(?:Access to model (\S+) is restricted|gated)", re.I),
     "{repo} is gated on Hugging Face: accept its terms on huggingface.co, log in "
     "(huggingface-cli login, or set HF_TOKEN), then python -m services setup stores the token"),
    (re.compile(r"GET failed \(404\)", re.I),
     "Hugging Face has no such model or quant - check the *_HF setting"),
    (re.compile(r"out of memory|cudaMalloc failed|failed to allocate", re.I),
     "it ran out of memory - fewer GPU layers (NGL), a smaller CTX, or EMBED_NGL/RERANK_NGL=0"),
)


def trouble(role, log=None):
    """A line saying why a role didn't come up, read from its log - or None."""
    if log is None:
        r = podman("logs", "--tail", "80", CONTAINER[role])
        log = r.stdout + r.stderr
    for pattern, hint in _TROUBLE:
        m = pattern.search(log)
        if m:
            repo = (m.group(1) if m.groups() and m.group(1) else "the model").rstrip(".")
            return hint.format(repo=repo)
    return None


# --- one role ---------------------------------------------------------------
def volume_find(image):
    """A search of the luna-models volume, for MATCH patterns."""
    def find(pattern):
        if podman("volume", "exists", VOLUME["models"]).returncode:
            return []
        out = podman("run", "--rm", "--entrypoint", "find", "-v", f"{VOLUME['models']}:/models:ro", image,
                     "/models", "-maxdepth", "5", "-iname", f"{pattern}.gguf", "!", "-iname", "*mmproj*").stdout
        return sorted(ln for ln in out.splitlines() if ln)
    return find


def plan(role, s, ctx, image_for=None):
    """(argv, env_text, model) for a role, or (None, why, None) to skip it.
    image_for(backend) gives an image ID, or None when there isn't one."""
    image_for = image_for or (lambda b: ctx.image_ids.get(b))
    any_image = next((i for i in ctx.image_ids.values() if i), None)
    model, why = settings.resolve(role, s, volume_find(any_image) if any_image else None)
    if model is None:
        return None, why, None
    backend = ctx.backend if run.wants_gpu(role, s, ctx.backend) else "cpu"
    image_id = image_for(backend)
    if not image_id:
        return None, f"no {backend} image yet - python -m services setup builds it", None
    env_text = settings.env_lines(role, s, model)
    env_file = os.path.join(env_dir(), f"{role}.env")
    argv = run.run_args(role, s, model, image=image_id, image_id=image_id, backend=backend,
                        env_file=env_file, env_text=env_text, key_fp=ctx.key_fp,
                        token_fp=ctx.token_fp, selinux=ctx.selinux, render=ctx.render,
                        cdi_dir=ctx.cdi_dir)
    return argv, env_text, model


def ensure(role, s, ctx, found=None):
    """Bring one llama role up, or keep it: a Result."""
    try:
        argv, env_text, model = plan(role, s, ctx)
    except settings.SettingsError as e:
        return Result(role, "failed", str(e))
    if argv is None:
        return Result(role, "skipped", env_text)
    return apply(role, argv, env_text, model.label(), settings.port(role, s), found)


def apply(role, argv, env_text, label, host_port, found=None):
    """Keep the role's container if it's current, else (re)make it - unless
    it isn't ours, or something else holds its port."""
    found = containers() if found is None else found
    name = CONTAINER[role]
    have = found.get(name)
    if have and "luna.config-hash" not in have["labels"]:
        return Result(role, "skipped", f"{name} is a container python -m services didn't make - "
                                       "python -m services setup asks before replacing it")
    want = argv[-2].split("=", 1)[1]
    if have and have["labels"].get("luna.config-hash") == want and have["state"] == "running":
        return Result(role, "kept", label)

    if not (have and have["state"] == "running"):
        holder = port_holder(host_port)
        if holder:
            return Result(role, "skipped", f"port {host_port} is taken by {holder} "
                                           "(a host llama/start.sh, LM Studio, a host voice?) - leaving it be")

    env_file = argv[argv.index("--env-file") + 1]
    with open(env_file, "w", encoding="utf-8") as f:
        f.write(env_text)
    r = subprocess.run(argv, capture_output=True, text=True)
    if r.returncode:
        last = (r.stderr.strip().splitlines() or ["podman run failed"])[-1]
        return Result(role, "failed", last)
    return Result(role, "started", label)


# --- the voice --------------------------------------------------------------
def voice_state():
    """The voice in install.json: how the installer set it up."""
    try:
        with open(os.path.join(ROOT, "install.json"), encoding="utf-8") as f:
            return (json.load(f) or {}).get("voice") or {}
    except (OSError, ValueError):
        return {}


def config_voice():
    """config.json's voice - the one she speaks with - for the server's default."""
    try:
        with open(os.path.join(ROOT, "config.json"), encoding="utf-8") as f:
            return str((json.load(f) or {}).get("voice") or "af_bella")
    except (OSError, ValueError):
        return "af_bella"


def default_roles(s=None):
    """What up starts when no roles are named: the llama roles, and the
    voice once setup tts has moved it into its container. With
    EMBED_ONDEMAND on, the embedder and reranker wait to be wanted."""
    roles = LLAMA_ROLES
    if s and str(s.get("EMBED_ONDEMAND", "")).lower() in ("on", "1", "true", "yes"):
        roles = ("chat",)
    return roles + (("tts",) if voice_state().get("container") == CONTAINER["tts"] else ())


def tts_device(s, backend):
    """The voice's device: cuda when TTS_DEVICE allows it and the machine is NVIDIA with CDI, else cpu."""
    want = (s.get("TTS_DEVICE") or "auto").lower()
    return "cuda" if want in ("auto", "cuda") and backend in images.NVIDIA else "cpu"


def ensure_tts(s, ctx, found=None, voice_mode=None):
    """The voice container, kept or (re)made: a Result. Never while the
    installer's host voice (managed) owns :8899."""
    mode = voice_state().get("mode") if voice_mode is None else voice_mode
    if mode == "managed":
        return Result("tts", "skipped", "her voice is the host's kokoro-reader (install.json: managed) - "
                                        "python -m services setup tts moves it into a container")
    device = tts_device(s, ctx.backend)
    build = images.tts_backend(ctx.backend) if device == "cuda" else "cpu"
    image_id = ctx.image_ids.get(("tts", build))
    if not image_id:
        return Result("tts", "skipped", f"no {build} voice image yet - python -m services setup tts builds it")
    voice = config_voice()
    env_text = f"KOKORO_DEVICE={device}\nKOKORO_VOICE={voice}\n"
    argv = run.tts_run_args(voice, image=image_id, image_id=image_id, device=device,
                            env_file=os.path.join(env_dir(), "tts.env"), env_text=env_text,
                            selinux=ctx.selinux, cdi_dir=ctx.cdi_dir)
    return apply("tts", argv, env_text, f"kokoro, {voice}, on the {device}", settings.TTS_PORT, found)


def up(s, roles=LLAMA_ROLES, token="", out=print):
    """Secrets current, then each role kept or (re)made. [Result]."""
    if any(role in LLAMA_ROLES for role in roles):
        # The voice takes no key: only the llama roles need these.
        if not keys.read_key():
            keys.make_key()
            out("Made llama/.api_key - Luna and the containers share it.")
        if keys.sync(keys.read_key(), token):
            out("Secrets updated - their containers are made anew.")
    ctx, note = context(s, token)
    if note:
        out("  " + note)
    ctx.image_ids = {b: images.find(b) for b in {ctx.backend, "cpu"}}
    ctx.image_ids.update({("tts", b): images.find(b, kind="tts") for b in {images.tts_backend(ctx.backend), "cpu"}})
    found = containers()
    results = [ensure(role, s, ctx, found) for role in roles if role in LLAMA_ROLES]
    if "tts" in roles:
        results.append(ensure_tts(s, ctx, found))
    return results


def down(roles=ROLES, found=None):
    """Stop and remove our containers (volumes and secrets stay). The
    names removed."""
    found = containers() if found is None else found
    gone = []
    for role in roles:
        name = CONTAINER[role]
        if name in found and "luna.config-hash" in found[name]["labels"]:
            podman("rm", "-f", "-t", "10", name)
            gone.append(name)
    return gone


def status(s, found=None, get=requests.get):
    """One line per role: its container, and whether it answers."""
    found = containers() if found is None else found
    lines = []
    for role in ROLES:
        name = CONTAINER[role]
        c = found.get(name)
        url = f"http://127.0.0.1:{settings.port(role, s)}"
        if not c:
            lines.append((False, f"{role:7} {name}: not created"))
            continue
        ours = "luna.config-hash" in c["labels"]
        try:
            code = get(url + "/health", timeout=2).status_code
        except requests.RequestException:
            code = None
        health = {200: "ready", 503: "loading"}.get(code, "not answering" if code is None else f"HTTP {code}")
        lines.append((code == 200 and ours, f"{role:7} {name}: {c['state']}, {health} on {url}"
                      + ("" if ours else "  (not made by python -m services)")))
    return lines


# --- reset ------------------------------------------------------------------
OUR_TAGS = ("localhost/luna-services-", "localhost/luna-deploy-")


def reset_plan(ps=None, imgs=None, vols=None, secrets=None):
    """What reset would remove: {"containers", "images", "volumes",
    "secrets", "models", "strangers"}. Only luna-* things: ours by label
    or our own names; luna-models (your downloads) and luna-* things some
    other setup made are listed apart, to be asked about on their own."""
    found = containers(ps)
    if imgs is None:
        imgs = json.loads(podman("images", "--format", "json").stdout or "[]")
    if vols is None:
        vols = podman("volume", "ls", "--format", "{{.Name}}").stdout.split()
    if secrets is None:
        secrets = podman("secret", "ls", "--format", "{{.Name}}").stdout.split()
    ours = sorted(n for n, c in found.items() if "luna.config-hash" in c["labels"])
    strangers = sorted(n for n, c in found.items() if "luna.config-hash" not in c["labels"])
    images_ours, images_strange = [], []
    for img in imgs:
        names = img.get("Names") or []
        labels = img.get("Labels") or {}
        ident = img.get("Id") or img.get("ID")
        if "luna.kind" in labels or any(n.startswith(OUR_TAGS) for n in names):
            images_ours.append(ident)
        elif any(n.startswith("localhost/luna-") and not n.startswith("localhost/luna-test-") for n in names):
            images_strange.append(ident)
    return {"containers": ours, "images": images_ours,
            "volumes": [v for v in (VOLUME["data"], VOLUME["tts"]) if v in vols],
            "secrets": [s for s in (keys.KEY_SECRET, keys.TOKEN_SECRET) if s in secrets],
            "models": [v for v in (VOLUME["models"],) if v in vols],
            "strangers": strangers, "strange_images": images_strange}


def reset(ask, out=print, plan=None):
    """Remove what python -m services made (asking first), and luna-models
    and other luna-* things only if asked separately. The names removed."""
    plan = plan or reset_plan()
    gone = []
    mine = plan["containers"] + plan["volumes"] + plan["secrets"]
    if mine or plan["images"]:
        out("  Ours: " + ", ".join(mine + [f"{len(plan['images'])} image(s)"]))
        if ask("Remove them?"):
            for name in plan["containers"]:
                podman("rm", "-f", "-t", "5", name)
            for ident in plan["images"]:
                podman("rmi", "-f", ident)
            for vol in plan["volumes"]:
                podman("volume", "rm", "-f", vol)
            for secret in plan["secrets"]:
                podman("secret", "rm", secret)
            gone += mine + plan["images"]
    if plan["models"] and ask(f"Remove {plan['models'][0]} too (the downloaded models - they'd download again)?",
                              default=False):
        podman("volume", "rm", "-f", plan["models"][0])
        gone.append(plan["models"][0])
    if plan["strangers"] or plan["strange_images"]:
        out("  Not made by python -m services: " + ", ".join(
            plan["strangers"] + [f"{len(plan['strange_images'])} image(s) named localhost/luna-*"]))
        if ask("Remove those as well?", default=False):
            for name in plan["strangers"]:
                podman("rm", "-f", "-t", "5", name)
            for ident in plan["strange_images"]:
                podman("rmi", "-f", ident)
            gone += plan["strangers"] + plan["strange_images"]
    return gone
