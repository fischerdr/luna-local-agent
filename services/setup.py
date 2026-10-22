"""python -m services setup: everything the containers need, once.

Asks before each step that touches the system, showing the exact
command first - images, volumes, secrets, files. Run it again any time:
it does only what's missing.
"""
import json
import os
import re
import shlex
import shutil
import subprocess
import sys

from services import (CONTAINER, KEY_SECRET, LLAMA_DIR, LLAMA_ROLES, ROOT, TOKEN_SECRET, HERE,
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


def pick_file(d, a):
    """One chat GGUF from a folder, by number. "" when there's none or
    nobody to ask (--yes doesn't guess which model you want)."""
    found = [p for p in settings.find_host("*", d)
             if not any(w in os.path.basename(p).lower() for w in ("embedding", "reranker"))]
    if not found or a.yes:
        return ""
    print("  GGUFs there:")
    for i, p in enumerate(found, 1):
        print(f"    {i}. {os.path.relpath(p, d)}  ({os.path.getsize(p) / 1e9:.1f} GB)")
    pick = a.text("Which is her chat model? (number, empty = none)")
    return found[int(pick) - 1] if pick.isdigit() and 1 <= int(pick) <= len(found) else ""


def side_models(s, a):
    """The embedding and reranker files already in LUNA_MODELS_DIR, used
    where they are instead of a second copy by -hf."""
    for role, hf_key, match_key in (("embed", "EMBED_HF", "EMBED_MATCH"), ("rerank", "RERANK_HF", "RERANK_MATCH")):
        if not s[hf_key] or not s[match_key]:
            continue
        found = settings.find_host(s[match_key], settings._models_dir(s))
        if len(found) == 1 and a.ask(f"{role}: use {os.path.basename(found[0])} from there (no download)?"):
            set_env(LOCAL_ENV, hf_key, "")
    return settings.load()


def models(s, a):
    """Step 8: a chat model the container can reach. Returns the settings,
    reread if they changed."""
    try:
        settings.resolve("chat", s)
        ok(f"chat model: {s['MODEL_HF'] or s['MODEL'] or s['MODEL_MATCH']}")
        return side_models(s, a) if s["LUNA_MODELS_DIR"] else s
    except settings.SettingsError as e:
        warn(str(e).splitlines()[0])
    for d in _gguf_dirs():
        if a.ask(f"Use the GGUFs in {d} (mounted read-only, no second copy)?"):
            set_env(LOCAL_ENV, "LUNA_MODELS_DIR", d)
            s = settings.load()
            try:
                settings.resolve("chat", s)
            except settings.SettingsError as e:
                warn(str(e).splitlines()[0])
                path = pick_file(d, a)
                if not path:
                    continue
                set_env(LOCAL_ENV, "MODEL", path)
                s = settings.load()
            ok(f"LUNA_MODELS_DIR={d} - in services/services.local.env")
            return side_models(s, a)
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
            sys.stdout.flush()
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

    if backend != "cpu" and a.ask("Measure where the reranker should run on this GPU (about a minute)?"):
        from services.bench import bench

        bench(a)
    autostart(a)
    print("  Next: python -m services up" + ("" if config_services().get("autostart") else ", or ./start.sh"))
    return True


def config_services():
    """config.json's "services" block (yours, never in git)."""
    try:
        with open(os.path.join(ROOT, "config.json"), encoding="utf-8") as f:
            return (json.load(f) or {}).get("services") or {}
    except (OSError, ValueError):
        return {}


def autostart(a):
    """Step 11: ./start.sh brings the containers up with her, and down after."""
    if config_services().get("autostart") is True:
        ok("./start.sh starts the containers with her (config.json services.autostart)")
        return
    if not a.ask("Start the containers with Luna when you run ./start.sh?"):
        return
    path = os.path.join(ROOT, "config.json")
    example = os.path.join(ROOT, "config.example.json")
    if not os.path.exists(path) and os.path.exists(example):
        shutil.copyfile(example, path)
    try:
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
    except (OSError, ValueError):
        warn("config.json couldn't be read - add \"services\": {\"autostart\": true} yourself")
        return
    cfg.setdefault("services", {})["autostart"] = True
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4, ensure_ascii=False)
        f.write("\n")
    ok("config.json: services.autostart - ./start.sh starts them, and stops them when she exits")


def save_voice(url):
    """Point install.json's voice at the container: launch.py then
    never starts the host's kokoro-reader as well. Other keys stay."""
    path = os.path.join(ROOT, "install.json")
    try:
        with open(path, encoding="utf-8") as f:
            state = json.load(f) or {}
    except (OSError, ValueError):
        state = {}
    state["voice"] = {"mode": "external", "url": url, "container": CONTAINER["tts"]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def setup_tts(a):
    """Set up her voice in a container (setup tts). True when it can come up."""
    print("Luna's voice in a container - setup")
    problem = lifecycle.podman_problem()
    if problem:
        warn(problem)
        print("  Without Podman, the installer sets her voice up on the host: ./installer/install.sh")
        return False
    s = settings.load()
    backend, _ = images.image_for_host(s["LLAMA_BACKEND"])
    device = lifecycle.tts_device(s, backend)
    build = images.tts_backend(backend) if device == "cuda" else "cpu"
    ok(f"voice on the {device}" + ("" if device == "cuda" else " - faster than real time"))

    if images.find(build, kind="tts"):
        ok(f"image {images.tag(build, 'tts')} is current")
    else:
        cmd = images.build_cmd(build, "tts")
        show(cmd)
        size = "about 1.6 GB" if build == "cpu" else "about 5 GB, CUDA's libraries included"
        if not a.ask(f"Build the voice image ({size})?"):
            return False
        sys.stdout.flush()
        if subprocess.call(cmd) != 0:
            warn("the build failed - the lines above say why")
            return False
        ok(f"built {images.tag(build, 'tts')}")

    if lifecycle.podman("volume", "exists", "luna-tts-cache").returncode:
        show(["podman", "volume", "create", "luna-tts-cache"])
        if a.ask("Create the luna-tts-cache volume (the voice model, about 330 MB, downloads into it once)?"):
            lifecycle.podman("volume", "create", "luna-tts-cache", check=True)

    v = lifecycle.voice_state()
    url = "http://127.0.0.1:8899"
    if v.get("container") != CONTAINER["tts"]:
        was = v.get("mode") or "not set up"
        if a.ask(f"Point install.json's voice at the container (it's {was} now), so ./start.sh "
                 "never starts the host's kokoro-reader too?"):
            save_voice(url)
            ok(f"install.json: voice external, {url}")
        else:
            warn("install.json left as it is - up tts won't run while it says managed")
    ok(f"voice: {lifecycle.config_voice()} (config.json)")
    print("  Next: python -m services up tts")
    return True


DEPLOY = os.path.join(ROOT, "deploy")
SERVER_PORTS = ("8080-8082", "8899")


def firewall_rules(client):
    """The commands that let one client in, for each firewall - printed,
    never run (they need root)."""
    import ipaddress

    family = "ipv6" if ipaddress.ip_address(client).version == 6 else "ipv4"
    nft = "ip6" if family == "ipv6" else "ip"
    firewalld = [f"sudo firewall-cmd --permanent --add-rich-rule='rule family={family} source address={client} "
                 f"port port={p} protocol=tcp accept'" for p in SERVER_PORTS] + ["sudo firewall-cmd --reload"]
    ufw = [f"sudo ufw allow from {client} to any port {p.replace('-', ':')} proto tcp" for p in SERVER_PORTS]
    nftables = [f"sudo nft add rule inet filter input {nft} saddr {client} tcp dport {{ 8080-8082, 8899 }} accept"]
    return {"firewalld": firewalld, "ufw": ufw, "nftables": nftables}


def compose_files(backend):
    """The -f files for this server's GPU, for podman compose and docker compose."""
    extra = {"cuda": "compose.cuda", "cuda13": "compose.cuda", "rocm": "compose.rocm",
             "vulkan": "compose.vulkan"}.get(backend)
    podman = ["-f", "compose.yaml"] + (["-f", f"{extra}.yaml"] if extra else [])
    docker = ["-f", "compose.yaml"] + (["-f", "compose.cuda.docker.yaml" if extra == "compose.cuda"
                                        else f"{extra}.yaml"] if extra else [])
    return podman, docker


def setup_server(a, allow=None):
    """Make deploy/ ready for compose (setup --server): this machine as a GPU
    server on your network. Prints the firewall rule; never runs sudo."""
    import ipaddress

    print("Luna's backend for the network - this machine as the server")
    if allow:
        try:
            ipaddress.ip_address(allow)
        except ValueError:
            warn(f"--allow {allow} isn't an IP address - the client's address on your network")
            return False

    key = os.path.join(DEPLOY, "api_key")
    if os.path.exists(key):
        ok("deploy/api_key is there")
    elif a.ask("Make deploy/api_key (random, readable only by you)?"):
        keys.make_key(key)
        ok("deploy/api_key - copy it into the client's llama/.api_key (the installer's --remote asks for it)")

    token = os.path.join(DEPLOY, "hf_token")
    if not os.path.exists(token):
        # compose won't start with a secret file missing; empty means none
        open(token, "w").close()
        os.chmod(token, 0o600)
        ok("deploy/hf_token, empty: public models only (put your token in it, or set HF_TOKEN_FILE in .env)")

    env = os.path.join(DEPLOY, ".env")
    if not os.path.exists(env) and a.ask("Make deploy/.env from .env.example?"):
        shutil.copyfile(os.path.join(DEPLOY, ".env.example"), env)
        ok("deploy/.env - set LUNA_BIND (this server's address) and MODEL_HF in it")

    backend, note = images.image_for_host("auto")
    if note:
        warn(note)
    podman, docker = compose_files(backend)
    print(f"  Start it ({backend}), from deploy/:")
    print("    podman compose " + " ".join(podman) + " up -d")
    print("    docker compose " + " ".join(docker) + " up -d")

    if allow:
        print(f"  Let {allow} in, and nothing else - for your firewall, one of:")
        for name, cmds in firewall_rules(allow).items():
            print(f"    {name}:")
            for cmd in cmds:
                print(f"      {cmd}")
    else:
        print("  Then let the client in: python -m services setup --server --allow <its IP> prints the rule.")
    print("  Plain HTTP on your LAN: the key guards chat, embeddings and the reranker; the voice has no key,")
    print("  so that firewall rule is what keeps everyone else out.")
    print("  On the client: ./installer/install.sh --remote <this server's address>")
    return True


if __name__ == "__main__":
    sys.exit(0 if setup(Asker("--yes" in sys.argv)) else 1)
