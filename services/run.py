"""The podman run for one role - a pure function, so tests can read it.

Every container:
- publishes on HOST (127.0.0.1 unless you share it) and nowhere else;
  never --network host;
- gets the key, and the token if there is one, as Podman secrets;
- keeps its state in named volumes: luna-models, and luna-data (saved
  chat slots);
- carries luna.config-hash, a SHA-256 of all of the above plus its
  settings, image and the secrets' fingerprints. A container with the
  same hash is kept as it is; any change makes a new one.
"""
import hashlib
import os

from services import CONTAINER, KEY_SECRET, TOKEN_SECRET
from services.images import NVIDIA
from services.settings import HOST_MODELS, port

HEALTH = "curl -fsS http://127.0.0.1:8080/health"


def gpu_flags(backend, selinux, render):
    """What hands the GPU to a container (proven on rootless Podman 5.8)."""
    if backend in NVIDIA:
        flags = ["--device", "nvidia.com/gpu=all"]
    elif backend == "rocm":
        flags = ["--device", "/dev/kfd", "--device", "/dev/dri", "--group-add", "video",
                 "--security-opt", "seccomp=unconfined"]
    elif backend == "vulkan":
        flags = ["--device", "/dev/dri", "--group-add", "video"]
    else:
        return []
    if render and backend in ("rocm", "vulkan"):
        flags += ["--group-add", "render"]
    if selinux:
        flags += ["--security-opt", "label=disable"]
    return flags


def publish(host, host_port):
    """-p's value: HOST only, IPv6 in brackets."""
    host = f"[{host}]" if ":" in host else host
    return f"{host}:{host_port}:8080"


def config_hash(argv, env_text, image_id, key_fp, token_fp):
    """What makes a container current: its argv, its settings, its image,
    and which key and token it holds (by fingerprint)."""
    h = hashlib.sha256()
    for part in (*argv, "\0env", env_text, "\0image", image_id, "\0key", key_fp, "\0token", token_fp):
        h.update(part.encode() + b"\0")
    return h.hexdigest()


def run_args(role, s, model, *, image, image_id, backend, env_file, env_text,
             key_fp, token_fp="", selinux=False, render=False, cdi_dir=None):
    """The full podman run argv for a llama role.

    backend is the image's (cpu for a role kept off the GPU); cdi_dir is
    set when the NVIDIA spec is a user-level one Podman doesn't look for."""
    argv = ["podman"]
    if cdi_dir and backend in NVIDIA:
        argv += ["--cdi-spec-dir", cdi_dir]
    argv += ["run", "-d", "--replace", "--name", CONTAINER[role],
             "--label", f"luna.role={role}",
             "-p", publish(s["HOST"], port(role, s)),
             "--secret", f"{KEY_SECRET},target=api_key"]
    if token_fp:
        argv += ["--secret", f"{TOKEN_SECRET},target=hf_token"]
    argv += ["-v", "luna-models:/models"]
    if role == "chat":
        argv += ["-v", "luna-data:/data"]
    if model.host_mount:
        # Read-only, and never relabelled: on SELinux the container is
        # let through instead (label=disable).
        argv += ["-v", f"{os.path.realpath(os.path.expanduser(s['LUNA_MODELS_DIR']))}:{HOST_MODELS}:ro"]
        if selinux and backend not in NVIDIA + ("rocm", "vulkan"):
            argv += ["--security-opt", "label=disable"]
    argv += ["--env-file", env_file,
             "--health-cmd", HEALTH, "--health-interval", "30s", "--health-start-period", "10m"]
    argv += gpu_flags(backend, selinux, render)
    h = config_hash(argv, env_text, image_id, key_fp, token_fp)
    return argv + ["--label", f"luna.config-hash={h}", image]


def wants_gpu(role, s, backend):
    """Whether a role offloads to the GPU at all: NGL 0 keeps it on the
    CPU, in the cpu image, so it takes no VRAM (a GPU build would still
    open a context there)."""
    ngl = {"chat": "NGL", "embed": "EMBED_NGL", "rerank": "RERANK_NGL"}[role]
    try:
        layers = int(s.get(ngl) or 0)
    except ValueError:
        layers = 99
    return backend != "cpu" and layers > 0
