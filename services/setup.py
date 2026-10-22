"""python -m services setup: everything the containers need, once.

Asks before each step that touches the system, showing the exact
command first - images, volumes, secrets, files. Run it again any time:
it does only what's missing.
"""
import os
import re
import shlex
import subprocess
import sys

from services import (CONTAINER, KEY_SECRET, LLAMA_DIR, LLAMA_ROLES, TOKEN_SECRET, HERE,
                      images, keys, lifecycle, settings)

LOCAL_ENV = os.path.join(HERE, "services.local.env")
LMSTUDIO_DIRS = ("~/.lmstudio/models", "~/.cache/lm-studio/models",
                 "~/.var/app/ai.lmstudio.lm-studio/.lmstudio/models")


class Asker:
    """Questions on the terminal; --yes takes every default."""

    def __init__(self, yes=False):
        self.yes = yes

    def ask(self, question, default=True):
        """A yes/no question; Enter (or no terminal) takes the default."""
        if self.yes:
            return default
        hint = "Y/n" if default else "y/N"
        try:
            answer = input(f"  {question} [{hint}] ").strip().lower()
        except EOFError:
            return default
        return default if not answer else answer.startswith("y")

    def text(self, question, default=""):
        """A question with a typed answer."""
        if self.yes:
            return default
        try:
            return input(f"  {question} ").strip() or default
        except EOFError:
            return default


def ok(msg):
    """A step that's done."""
    print(f"  ✓ {msg}")


def warn(msg):
    """A step that needs you."""
    print(f"  ! {msg}")


def show(cmd):
    """The exact command, before asking to run it."""
    print("    $ " + " ".join(shlex.quote(c) for c in cmd), flush=True)


def set_env(path, key, value):
    """Set KEY="value" in one of the .local.env files - replacing its line
    if it has one, else adding it. Nothing else in the file changes."""
    line = f'{key}="{value}"'
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except OSError:
        text = "# This machine's own settings, read after services.env - not committed.\n"
    pattern = re.compile(rf"^{re.escape(key)}=.*$", re.M)
    text = pattern.sub(line, text, count=1) if pattern.search(text) else text.rstrip("\n") + "\n" + line + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _gguf_dirs():
    """Folders here that already hold GGUFs: llama/models, LM Studio's."""
    found = []
    for d in (os.path.join(LLAMA_DIR, "models"), *(os.path.expanduser(p) for p in LMSTUDIO_DIRS)):
        if os.path.isdir(d) and settings.find_host("*", d) + settings.find_host("*", d, mmproj=True):
            found.append(d)
    return found


def models(s, a):
    """Step 8: a chat model the container can reach. Returns the settings,
    reread if they changed."""
    try:
        settings.resolve("chat", s)
        ok(f"chat model: {s['MODEL_HF'] or s['MODEL'] or s['MODEL_MATCH']}")
        return s
    except settings.SettingsError as e:
        warn(str(e).splitlines()[0])
    for d in _gguf_dirs():
        if a.ask(f"Use the GGUFs in {d} (mounted read-only, no second copy)?"):
            set_env(LOCAL_ENV, "LUNA_MODELS_DIR", d)
            s = settings.load()
            try:
                settings.resolve("chat", s)
                ok(f"LUNA_MODELS_DIR={d} - in services/services.local.env")
                return s
            except settings.SettingsError as e:
                warn(str(e))
    hf = a.text("A model from Hugging Face instead, as <user>/<repo>:<quant> (empty = skip):")
    if hf:
        set_env(LOCAL_ENV, "MODEL_HF", hf)
        ok(f"MODEL_HF={hf} - downloaded on the first up, into the luna-models volume")
        return settings.load()
    warn("no chat model yet - set MODEL_HF or LUNA_MODELS_DIR in services/services.local.env")
    return s


def setup(a, gpu="auto"):
    """The steps, in order. True when the containers can come up."""
    print("Luna's backend in containers - setup")

    problem = lifecycle.podman_problem()
    if problem:
        warn(problem)
        print("  Without Podman: LM Studio, or llama.cpp built here (bash llama/install.sh).")
        return False
    ok("Podman, rootless")

    s = settings.load()
    if gpu != "auto":
        s["LLAMA_BACKEND"] = gpu
    backend, note = images.image_for_host(s["LLAMA_BACKEND"])
    ok(f"image: {backend}")
    if note:
        warn(note)

    # The key: llama/.api_key, shared with Luna and llama/start.sh.
    key = keys.read_key()
    if not key:
        if not a.ask("Make llama/.api_key (random, readable only by you)?"):
            warn("no key - the containers won't start without one")
            return False
        key = keys.make_key()
    token, where = keys.token_source()
    print(f"    $ podman secret create --replace {KEY_SECRET} -   (and {TOKEN_SECRET}, on stdin)")
    if a.ask("Store the key (and your Hugging Face token, if any) as Podman secrets?"):
        keys.sync(key, token)
        tail = (f", and {TOKEN_SECRET} from {where}" if token
                else " - no Hugging Face token, so public models only")
        ok(f"secret {KEY_SECRET}{tail}")

    # Containers some other setup made, with these names.
    strangers = lifecycle.unlabelled()
    if strangers:
        warn("these containers weren't made by python -m services: " + ", ".join(strangers))
        mine = [n for n in strangers if n in CONTAINER.values()]
        if mine and a.ask(f"Stop and remove {', '.join(mine)}, so ours can take their place?", default=False):
            for name in mine:
                show(["podman", "rm", "-f", name])
                lifecycle.podman("rm", "-f", name)
            ok("removed")
        elif mine:
            warn("left as they are - up skips those roles")

    # Images: this machine's backend, and cpu for roles kept off the GPU.
    for b in dict.fromkeys((backend, "cpu")):
        if images.find(b):
            ok(f"image {images.tag(b)} is current")
            continue
        cmd = images.build_cmd(b)
        show(cmd)
        if a.ask(f"Build the {b} image (pulls upstream's llama.cpp, about 1-4 GB)?"):
            if subprocess.call(cmd) != 0:
                warn("the build failed - the lines above say why")
                return False
            ok(f"built {images.tag(b)}")

    for vol in ("luna-models", "luna-data"):
        if lifecycle.podman("volume", "exists", vol).returncode == 0:
            continue
        show(["podman", "volume", "create", vol])
        if a.ask(f"Create the {vol} volume?"):
            lifecycle.podman("volume", "create", vol, check=True)
    ok("volumes luna-models, luna-data")

    s = models(s, a)
    for role in LLAMA_ROLES[1:]:
        try:
            m, why = settings.resolve(role, s)
            ok(f"{role}: {m.label()}" if m else f"{role}: skipped - {why}")
        except settings.SettingsError as e:
            warn(str(e))

    print("  Next: python -m services up")
    return True


if __name__ == "__main__":
    sys.exit(0 if setup(Asker("--yes" in sys.argv)) else 1)
