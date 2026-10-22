"""The containers' settings, and which model file each role gets.

Settings are llama/start.sh's own, read the way it reads them - by bash,
never a parser of ours - plus the few the containers add:

    llama/server.env        (or LLAMA_ENV=<file>, which replaces it)
    llama/server.local.env  (if there)
    services/services.env
    services/services.local.env  (if there)

Each later file wins. Only the names below are kept.

Models are settled here, on the host, before a container starts - never
searched for at runtime. Per role, the first of:

1. *_HF set: llama-server's -hf downloads it into the luna-models volume.
2. MODEL (EMBED_MODEL, RERANK_MODEL) names a file: one under
   LUNA_MODELS_DIR becomes /models-host/<the same relative path>; one
   starting /models/ is already in the volume; anything else is an
   error that says which folder to mount.
3. MODEL_MATCH (EMBED_MATCH, RERANK_MATCH): start.sh's search, in
   LUNA_MODELS_DIR then the volume. More than one match is an error.
4. Nothing: chat refuses; embed and rerank are skipped when their switch
   is auto (refused when it's on), as start.sh does.
"""
import fnmatch
import os
import subprocess
from dataclasses import dataclass

from services import HERE, LLAMA_DIR

# llama/server.env's names - start.sh's switches stay the switches.
SERVER_KEYS = (
    "MODEL", "MODEL_MATCH", "MMPROJ", "MMPROJ_MATCH", "ALIAS", "HOST", "PORT",
    "CTX", "NGL", "THREADS", "BATCH", "UBATCH", "PARALLEL", "KV_UNIFIED", "CTX_CHECKPOINTS",
    "REASONING_FORMAT", "REASONING_BUDGET", "EXTRA",
    "EMBED", "EMBED_MODEL", "EMBED_MATCH", "EMBED_PORT", "EMBED_CTX", "EMBED_POOLING",
    "RERANK", "RERANK_MODEL", "RERANK_MATCH", "RERANK_PORT", "RERANK_CTX",
)
# services/services.env's - what only the containers need.
SERVICE_KEYS = (
    "MODEL_HF", "MMPROJ_HF", "EMBED_HF", "RERANK_HF", "LUNA_MODELS_DIR",
    "EMBED_NGL", "RERANK_NGL", "LLAMA_BACKEND", "READY_TIMEOUT",
)
KNOWN = SERVER_KEYS + SERVICE_KEYS

# start.sh's own fallbacks, for names an older server.env doesn't have.
DEFAULTS = {"HOST": "127.0.0.1", "PORT": "8080", "EMBED": "auto", "EMBED_PORT": "8081",
            "RERANK": "auto", "RERANK_PORT": "8082", "RERANK_CTX": "2048",
            "RERANK_MATCH": "*qwen3-reranker*"}

# The env files, sourced in start.sh's order: $1 and $2 are services.env
# and services.local.env. A relative LLAMA_ENV is relative to llama/, as
# in start.sh. env is found first: a file may well set PATH. set -e: a
# file bash can't read is an error, not a silent half.
_READ = r'''set -e
_env="$(command -v env)"
set -a
case "${LLAMA_ENV:-server.env}" in /*) f="$LLAMA_ENV" ;; *) f="./${LLAMA_ENV:-server.env}" ;; esac
source "$f"
if [ -f server.local.env ]; then source ./server.local.env; fi
source "$1"
if [ -f "$2" ]; then source "$2"; fi
"$_env" -0'''

PORT_KEY = {"chat": "PORT", "embed": "EMBED_PORT", "rerank": "RERANK_PORT"}
TTS_PORT = 8899

# Per role: its -hf name, its file, its search pattern, and its switch.
_ROLE_KEYS = {
    "chat": ("MODEL_HF", "MODEL", "MODEL_MATCH", None),
    "embed": ("EMBED_HF", "EMBED_MODEL", "EMBED_MATCH", "EMBED"),
    "rerank": ("RERANK_HF", "RERANK_MODEL", "RERANK_MATCH", "RERANK"),
}

# What each role's container is told, besides its model.
PASS = {
    "chat": ("ALIAS", "CTX", "NGL", "THREADS", "BATCH", "UBATCH", "PARALLEL", "KV_UNIFIED",
             "CTX_CHECKPOINTS", "REASONING_FORMAT", "REASONING_BUDGET", "EXTRA"),
    "embed": ("THREADS", "EMBED_CTX", "EMBED_POOLING", "EMBED_NGL"),
    "rerank": ("THREADS", "RERANK_CTX", "RERANK_NGL"),
}

HOST_MODELS = "/models-host"      # LUNA_MODELS_DIR, read-only
VOLUME_MODELS = "/models"         # the luna-models volume


class SettingsError(Exception):
    """A setting that can't work, in words that say how to fix it."""


def load(llama_dir=LLAMA_DIR, services_dir=HERE, llama_env=None):
    """Every known setting, resolved by bash from the four env files."""
    llama_env = os.environ.get("LLAMA_ENV", "") if llama_env is None else llama_env
    env = {"HOME": os.path.expanduser("~"), "PATH": "/usr/bin:/bin"}
    if llama_env:
        env["LLAMA_ENV"] = llama_env
    r = subprocess.run(["bash", "-c", _READ, "services-settings",
                        os.path.join(services_dir, "services.env"),
                        os.path.join(services_dir, "services.local.env")],
                       cwd=llama_dir, env=env, capture_output=True, text=True)
    if r.returncode:
        raise SettingsError("couldn't read the settings: " + (r.stderr.strip() or f"bash exit {r.returncode}"))
    found = dict(kv.partition("=")[::2] for kv in r.stdout.split("\0") if "=" in kv)
    s = dict(DEFAULTS)
    s.update({k: v for k, v in found.items() if k in KNOWN})
    for k in KNOWN:
        s.setdefault(k, "")
    return s


def port(role, s):
    """The host port a role answers on."""
    return TTS_PORT if role == "tts" else int(s[PORT_KEY[role]])


def host_warning(s):
    """A line to say when HOST shares the servers beyond this machine."""
    if s["HOST"] in ("127.0.0.1", "localhost", "::1"):
        return None
    return (f"HOST is {s['HOST']} in llama/server.env, so the containers answer other machines "
            "too - anyone who can reach them still needs the key.")


@dataclass
class Model:
    """Where a role's model comes from, as the container sees it."""

    hf: str = ""             # <user>/<repo>[:quant] for -hf
    file: str = ""           # a path inside the container
    mmproj: str = ""         # chat: an mmproj file inside the container
    mmproj_url: str = ""     # chat: an mmproj to download (MMPROJ_HF)
    no_mmproj: bool = False  # chat: MMPROJ=off
    host_mount: bool = False  # needs LUNA_MODELS_DIR mounted

    def label(self):
        """How the model is named in what we print."""
        return self.hf or os.path.basename(self.file)


def _models_dir(s):
    d = s["LUNA_MODELS_DIR"]
    return os.path.realpath(os.path.expanduser(d)) if d else ""


def map_path(path, s):
    """A model path from the settings, as the container sees it."""
    if path.startswith(VOLUME_MODELS + "/"):
        return path
    real = os.path.realpath(os.path.expanduser(path))
    root = _models_dir(s)
    if root and (real + os.sep).startswith(root + os.sep):
        if not os.path.isfile(real):
            raise SettingsError(f"model file not found: {path}")
        return HOST_MODELS + "/" + os.path.relpath(real, root).replace(os.sep, "/")
    folder = os.path.dirname(real)
    raise SettingsError(f"{path} isn't in a folder the containers can see - set "
                        f"LUNA_MODELS_DIR=\"{folder}\" (or a folder above it) in services/services.local.env")


def find_host(pattern, root, mmproj=False):
    """start.sh's search: case-insensitive, five levels down, mmproj files
    skipped (or, for an mmproj, required). Paths under root, sorted."""
    if not root or not os.path.isdir(root):
        return []
    want = (pattern + ".gguf").lower()
    base = root.rstrip(os.sep).count(os.sep)
    found = []
    for d, dirs, files in os.walk(root, followlinks=True):
        if d.count(os.sep) - base >= 5:
            dirs[:] = []
        for f in files:
            name = f.lower()
            if fnmatch.fnmatchcase(name, want) and ("mmproj" in name) == mmproj:
                found.append(os.path.join(d, f))
    return sorted(found)


def _one(found, pattern):
    if len(found) > 1:
        raise SettingsError(f"more than one model matches {pattern} - set the path instead:\n  "
                            + "\n  ".join(found))
    return found[0] if found else ""


def resolve(role, s, volume_find=None):
    """(Model, None) for a role that runs, or (None, why) for one that's
    skipped. Raises SettingsError when it must run and can't.

    volume_find(pattern) lists matching files already in the volume."""
    hf_key, file_key, match_key, switch_key = _ROLE_KEYS[role]
    switch = (s[switch_key] or "auto").lower() if switch_key else "on"
    if switch == "off":
        return None, f"off ({switch_key}=off in llama/server.env)"

    m, why = Model(), ""
    if s[hf_key]:
        m.hf = s[hf_key]
    elif s[file_key]:
        m.file = map_path(s[file_key], s)
    elif s[match_key]:
        pattern = s[match_key]
        root = _models_dir(s)
        host = _one(find_host(pattern, root), pattern)
        if host:
            m.file = map_path(host, s)
        elif volume_find:
            m.file = _one(volume_find(pattern), pattern)
        if not m.file:
            where = "LUNA_MODELS_DIR or the luna-models volume" if root else "the luna-models volume (LUNA_MODELS_DIR isn't set)"
            why = f"no GGUF matching {pattern} in {where}"
    else:
        why = f"no model set ({hf_key}, {file_key} or {match_key})"

    if not (m.hf or m.file):
        if role == "chat":
            raise SettingsError(f"{why} - set MODEL_HF in services/services.local.env, or "
                                "LUNA_MODELS_DIR to the folder with your GGUF")
        if switch == "on":
            raise SettingsError(f"{role}: {why}, and {switch_key}=on in llama/server.env")
        return None, why

    m.host_mount = m.file.startswith(HOST_MODELS + "/")
    if role == "chat":
        _vision(m, s)
    return m, None


def _vision(m, s):
    """The chat model's mmproj, by start.sh's rules (see Model)."""
    mmproj = s["MMPROJ"]
    if mmproj == "off":
        m.no_mmproj = True
    elif s["MMPROJ_HF"]:
        repo, _, name = s["MMPROJ_HF"].rpartition("/")
        if not repo or "/" not in repo:
            raise SettingsError(f"MMPROJ_HF should be <user>/<repo>/<file>.gguf, not {s['MMPROJ_HF']}")
        m.mmproj_url = f"https://huggingface.co/{repo}/resolve/main/{name}"
    elif mmproj:
        m.mmproj = map_path(mmproj, s)
    elif m.host_mount:
        # Beside the model first (LM Studio puts it there), then by MMPROJ_MATCH.
        root = _models_dir(s)
        folder = os.path.join(root, os.path.dirname(m.file[len(HOST_MODELS) + 1:]))
        found = [os.path.join(folder, f) for f in sorted(os.listdir(folder))
                 if f.lower().endswith(".gguf") and "mmproj" in f.lower()]
        if not found and s["MMPROJ_MATCH"]:
            found = find_host(s["MMPROJ_MATCH"], root, mmproj=True)
        if len(found) > 1:
            raise SettingsError("more than one mmproj found - set MMPROJ= in llama/server.local.env:\n  "
                                + "\n  ".join(found))
        if found:
            m.mmproj = map_path(found[0], s)
    # An -hf model brings its own mmproj, if its repo has one.


def env_lines(role, s, m):
    """The container's --env-file: its role, its model, and the settings
    it uses. No secrets - those are Podman secrets."""
    out = [("ROLE", role)]
    if m.hf:
        out.append(("LUNA_HF", m.hf))
    else:
        out.append(("LUNA_MODEL", m.file))
    if m.no_mmproj:
        out.append(("MMPROJ", "off"))
    elif m.mmproj:
        out.append(("LUNA_MMPROJ", m.mmproj))
    elif m.mmproj_url:
        out.append(("LUNA_MMPROJ_URL", m.mmproj_url))
    for k in PASS[role]:
        if s.get(k, "") != "" or k == "EMBED_POOLING":
            out.append((k, s.get(k, "")))
    for k, v in out:
        if "\n" in v or "\0" in v:
            raise SettingsError(f"{k} can't hold a line break")
    return "".join(f"{k}={v}\n" for k, v in out)
