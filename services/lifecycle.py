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

from services import CONTAINER, LLAMA_ROLES, MIN_PODMAN, ROLES, images, keys, run, settings


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
        if name.startswith("luna-") and not name.startswith("luna-test-"):
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


def wait_ready(url, timeout, get=requests.get, clock=time.monotonic, pause=time.sleep):
    """Poll url/health until it says ok (200; 503 means still loading).
    True when ready, False when the time ran out."""
    end = clock() + timeout
    while True:
        try:
            if get(url.rstrip("/") + "/health", timeout=2).status_code == 200:
                return True
        except requests.RequestException:
            pass
        if clock() >= end:
            return False
        pause(1)


# --- one role ---------------------------------------------------------------
def volume_find(image):
    """A search of the luna-models volume, for MATCH patterns."""
    def find(pattern):
        if podman("volume", "exists", "luna-models").returncode:
            return []
        out = podman("run", "--rm", "--entrypoint", "find", "-v", "luna-models:/models:ro", image,
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
    found = containers() if found is None else found
    try:
        argv, env_text, model = plan(role, s, ctx)
    except settings.SettingsError as e:
        return Result(role, "failed", str(e))
    if argv is None:
        return Result(role, "skipped", env_text)

    name = CONTAINER[role]
    have = found.get(name)
    if have and "luna.config-hash" not in have["labels"]:
        return Result(role, "skipped", f"{name} is a container python -m services didn't make - "
                                       "python -m services setup asks before replacing it")
    want = argv[-2].split("=", 1)[1]
    if have and have["labels"].get("luna.config-hash") == want and have["state"] == "running":
        return Result(role, "kept", model.label())

    if not (have and have["state"] == "running"):
        holder = port_holder(settings.port(role, s))
        if holder:
            return Result(role, "skipped", f"port {settings.port(role, s)} is taken by {holder} "
                                           "(a host llama/start.sh, LM Studio?) - leaving it be")

    env_file = argv[argv.index("--env-file") + 1]
    with open(env_file, "w", encoding="utf-8") as f:
        f.write(env_text)
    r = subprocess.run(argv, capture_output=True, text=True)
    if r.returncode:
        last = (r.stderr.strip().splitlines() or ["podman run failed"])[-1]
        return Result(role, "failed", last)
    return Result(role, "started", model.label())


def up(s, roles=LLAMA_ROLES, token="", out=print):
    """Secrets current, then each role kept or (re)made. [Result]."""
    if not keys.read_key():
        keys.make_key()
        out("Made llama/.api_key - Luna and the containers share it.")
    if keys.sync(keys.read_key(), token):
        out("Secrets updated - their containers are made anew.")
    ctx, note = context(s, token)
    if note:
        out("  " + note)
    ctx.image_ids = {b: images.find(b) for b in {ctx.backend, "cpu"}}
    found = containers()
    return [ensure(role, s, ctx, found) for role in roles if role in LLAMA_ROLES]


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
