"""Which llama.cpp image this machine gets, and building ours on it.

image_for_host() picks the backend from the GPU it finds:

1. NVIDIA, with a CDI spec: cuda13 when the driver's CUDA is 13 or
   newer, else cuda.
2. NVIDIA without one: cpu, and the command that makes the spec. Never
   Vulkan: on NVIDIA that needs the vendor's ICD inside the container.
3. /dev/kfd (AMD ROCm): rocm on x86_64; vulkan on arm64, which has no
   rocm image.
4. /dev/dri: vulkan.
5. Nothing: cpu.

LLAMA_BACKEND in services.env (or --gpu) overrides it.

Our images are known by their labels, never by tag alone: an old
localhost/luna-llama:cuda13 retag from somewhere else is not ours.
"""
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess

from services import HERE

LLAMA_CONTEXT = os.path.join(HERE, "llama")
REPO = "ghcr.io/ggml-org/llama.cpp"

# Upstream's images, pinned (multi-arch index digests, build b11804).
# Tags move; a new pin is a new image-version, so ensure() rebuilds.
BASES = {
    "cpu": ("server", "sha256:8c74db29f1a2bef3f0a331b52c4d6214601dfceed4bee0b94cc48b3b23ff257d"),
    "cuda": ("server-cuda", "sha256:490d8cd8000203a3ab8fdfce14ef931e2aa85affd4dd4447bcc75f299f00fd9a"),
    "cuda13": ("server-cuda13", "sha256:dfe42bda81c6bd53df864dfcf65e2423e9703f9991c238cfa7aaa893063e2977"),
    "rocm": ("server-rocm", "sha256:bdbbab3b239f4a7ada461ddb82b01d7328c6988e194572ae4f08144b710f1013"),
    "vulkan": ("server-vulkan", "sha256:a644fb651975d9dea49140302577a3b945993fd5efd3400bf89c7d525d03c103"),
}
NVIDIA = ("cuda", "cuda13")

CDI_SYSTEM = "/etc/cdi/nvidia.yaml"
CDI_USER = os.path.expanduser("~/.config/cdi/nvidia.yaml")
CDI_COMMAND = f"nvidia-ctk cdi generate --output={CDI_USER}"


def cuda_major():
    """The driver's CUDA version (nvidia-smi's header), or None without one."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"CUDA Version:\s*(\d+)", out)
    return int(m.group(1)) if m else None


def cdi_spec():
    """The NVIDIA CDI spec Podman will read, or None."""
    for path in (CDI_SYSTEM, CDI_USER):
        if os.path.exists(path):
            return path
    return None


def image_for_host(override="auto", cuda=cuda_major, cdi=cdi_spec, exists=os.path.exists,
                   arch=platform.machine):
    """(backend, note): the backend to use here, and a line to say about it
    (the CDI command, an override that can't work), or None."""
    override = (override or "auto").lower()
    major = cuda()
    if override != "auto":
        if override not in BASES:
            return "cpu", f"LLAMA_BACKEND={override} isn't one of {', '.join(BASES)} - using cpu"
        if override in NVIDIA and major is None:
            return "cpu", f"{override} asked for, but there's no NVIDIA GPU here - using cpu"
        if override == "rocm" and not exists("/dev/kfd"):
            return "cpu", "rocm asked for, but there's no /dev/kfd here - using cpu"
        return override, None
    if major is not None:
        if cdi():
            return ("cuda13" if major >= 13 else "cuda"), None
        return "cpu", ("an NVIDIA GPU, but Podman can't hand it to a container yet - make its CDI spec, "
                       f"then run setup again:\n    {CDI_COMMAND}\n  (if that needs root: "
                       f"sudo nvidia-ctk cdi generate --output={CDI_SYSTEM})")
    if exists("/dev/kfd"):
        return ("vulkan" if arch() in ("aarch64", "arm64") else "rocm"), None
    if exists("/dev/dri"):
        return "vulkan", None
    return "cpu", None


def image_version():
    """Changes whenever the image would: our build files, or the pins."""
    h = hashlib.sha256(json.dumps(BASES, sort_keys=True).encode())
    for name in sorted(os.listdir(LLAMA_CONTEXT)):
        with open(os.path.join(LLAMA_CONTEXT, name), "rb") as f:
            h.update(name.encode() + b"\0" + f.read())
    return h.hexdigest()[:16]


def labels(backend):
    """The labels a current image of ours carries."""
    return {"luna.image-version": image_version(), "luna.base-digest": BASES[backend][1],
            "luna.backend": backend}


def tag(backend):
    """The local tag of our image for a backend (a name, not proof - see
    find). Not luna-llama: an older setup used that name, and a build here
    mustn't move its tag."""
    return f"localhost/luna-services-llama:{backend}"


def build_cmd(backend):
    """The podman build that makes our image for a backend."""
    base, digest = BASES[backend]
    cmd = ["podman", "build", "--build-arg", f"BACKEND={base}", "--build-arg", f"BASE_DIGEST={digest}"]
    for k, v in labels(backend).items():
        cmd += ["--label", f"{k}={v}"]
    return cmd + ["-t", tag(backend), LLAMA_CONTEXT]


def find(backend, images=None):
    """The ID of a local image of ours for this backend, current by its
    labels - or None. A tag alone proves nothing."""
    want = labels(backend)
    if images is None:
        out = subprocess.run(["podman", "images", "--format", "json"], capture_output=True, text=True).stdout
        images = json.loads(out or "[]")
    for img in images:
        got = img.get("Labels") or {}
        if all(got.get(k) == v for k, v in want.items()):
            return img.get("Id") or img.get("ID")
    return None
