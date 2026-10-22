"""llama.cpp container management via Podman.

Uses pre-built llama.cpp images from ghcr.io/ggml-org/llama.cpp:
  - server-cuda13   CUDA 13 (NVIDIA RTX 30xx/40xx, driver >= 580)
  - server-cuda     CUDA 12 (older NVIDIA)
  - server-vulkan   Vulkan (AMD, Intel, other)
  - server-rocm     ROCm (AMD CDNA/RDNA)
  - server          CPU only

The container is named "luna-llama", port 8080.

Called by:
  - install.py at install time (creates/pulls the image)
  - start.sh at runtime (starts/stops the container)
  - launch.py at app shutdown (stops the container)
"""
import json
import os
import subprocess


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_NAME = "luna-llama"
CONTAINER_NAME = "luna-llama"
LLAMA_DIR = os.path.join(ROOT, "llama")
MODEL_DIR = os.path.join(LLAMA_DIR, "models")
CONFIG_DIR = LLAMA_DIR

# Pre-built image selection based on GPU
_CUDA13_IMAGE = f"{IMAGE_NAME}:cuda13"
_CUDA_IMAGE = f"{IMAGE_NAME}:cuda"
_VULKAN_IMAGE = f"{IMAGE_NAME}:vulkan"
_ROCM_IMAGE = f"{IMAGE_NAME}:rocm"
_CPU_IMAGE = f"{IMAGE_NAME}:cpu"


def _has_nvidia_gpu():
    """Check if an NVIDIA GPU is available on the host."""
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _has_cdi_nvidia():
    """Check if NVIDIA CDI device is available for Podman.
    
    CDI requires one-time host setup with:
      sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
    
    Returns True if the CDI spec file exists.
    """
    cdi_paths = [
        "/etc/cdi/nvidia.yaml",
        "/etc/cdi/nvidia.yml",
    ]
    # Also check user-level CDI directory
    home_cdi = os.path.expanduser("~/.config/cdi")
    if os.path.isdir(home_cdi):
        cdi_paths.append(os.path.join(home_cdi, "nvidia.yaml"))
    
    for path in cdi_paths:
        if os.path.isfile(path):
            return True
    return False


def _has_rocm_gpu():
    """Check if an AMD GPU with ROCm support is available."""
    try:
        # Check for ROCm device nodes
        return os.path.exists("/dev/kfd") and os.path.exists("/dev/dri")
    except Exception:
        return False


def _cuda_version():
    """Check CUDA driver version from nvidia-smi. Returns major version."""
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            # "CUDA Version: 13.0"
            for line in result.stdout.splitlines():
                if "CUDA Version" in line:
                    ver = line.split(":")[-1].strip()
                    return int(ver.split(".")[0])
    except Exception:
        pass
    return 0


def _detect_llama_image():
    """Select the best pre-built image for the current GPU."""
    import platform
    if _has_nvidia_gpu():
        if _cuda_version() >= 13:
            return _CUDA13_IMAGE
        return _CUDA_IMAGE
    if platform.system() == "Darwin":
        # macOS: use CPU image (Metal support in base image)
        return _CPU_IMAGE
    # Fall back to Vulkan (works on AMD, Intel, most Linux GPUs)
    return _VULKAN_IMAGE


def build_llama_container(mode=None):
    """Build or pull the llama.cpp container image.
    
    For NVIDIA GPUs, uses the CUDA pre-built image from GHCR.
    For others, pulls the Vulkan pre-built image.
    
    Args:
        mode: "cuda", "cuda13", "vulkan", "rocm", "cpu".
              If None, auto-detected from GPU.
    
    Returns:
        True on success, False on failure.
    """
    if mode is None:
        mode = _detect_llama_image().split(":")[-1]
    
    image_tag = f"{IMAGE_NAME}:{mode}"
    
    # Check if image already exists locally
    result = subprocess.run(
        ["podman", "images", "-q", image_tag],
        capture_output=True, text=True
    )
    if result.stdout.strip():
        return True  # Already built/pulled
    
    # Try pulling pre-built image from GHCR first
    pull_cmd = ["podman", "pull", f"ghcr.io/ggml-org/llama.cpp:server-{mode}"]
    result = subprocess.run(pull_cmd, capture_output=True, text=True)
    if result.returncode == 0:
        # Tag the pulled image
        subprocess.run(
            ["podman", "tag", f"ghcr.io/ggml-org/llama.cpp:server-{mode}", image_tag],
            capture_output=True, text=True
        )
        return True
    
    # Fall back to building from Dockerfile if pull fails
    dockerfile = os.path.join(LLAMA_DIR, "Dockerfile.cuda")
    if os.path.isfile(dockerfile):
        result = subprocess.run(
            ["podman", "build", "-t", image_tag, "-f", dockerfile, ROOT],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            return True
    
    return False


def start_llama_container():
    """Start the llama.cpp container if not already running. Idempotent.
    
    Mounts:
        /models  - model files (read-only)
        /config  - server.env, server.local.env (read-only)
        /data    - writable (slots, .api_key)
    
    Returns:
        True if container is running, False on failure.
    """
    image_tag = _detect_llama_image()
    
    # Check if container already running
    result = subprocess.run(
        ["podman", "ps", "-a", "--filter", f"name={CONTAINER_NAME}",
         "--format", "{{.Status}}"],
        capture_output=True, text=True
    )
    status = result.stdout.strip()
    
    if "Up" in status:
        return True
    
    if "Exited" in status or "exited" in status:
        # Restart stopped container
        subprocess.run(
            ["podman", "start", CONTAINER_NAME],
            check=False, capture_output=True
        )
        return True
    
    # Build image if needed
    if not build_llama_container():
        return False
    
    # Remove old container if it exists with conflicting name
    subprocess.run(
        ["podman", "rm", "-f", CONTAINER_NAME],
        check=False, capture_output=True
    )
    
    # Create writable data dir and required files (Podman requires host dir to exist)
    data_dir = os.path.join(LLAMA_DIR, "data")
    slots_dir = os.path.join(data_dir, "slots")
    os.makedirs(slots_dir, exist_ok=True)
    api_key = os.path.join(data_dir, ".api_key")
    if not os.path.isfile(api_key):
        with open(api_key, "w") as f:
            f.write("".join(chr(ord("a") + (i % 26)) for i in range(64)))

    # Find the model to pass explicitly (auto-detection doesn't work)
    server_env = os.path.join(CONFIG_DIR, "server.env")
    model = ""
    model_match = ""
    if os.path.isfile(server_env):
        with open(server_env) as f:
            for line in f:
                line = line.strip()
                if line.startswith("MODEL="):
                    model = line.split("=", 1)[1].strip().strip('"').strip("'")
                elif line.startswith("MODEL_MATCH="):
                    model_match = line.split("=", 1)[1].strip().strip('"').strip("'")

    # If no explicit model path, scan for a matching GGUF in the mounted models dir
    if not model and model_match:
        # Expand the glob pattern to find a matching file
        import fnmatch
        ggufs = os.listdir(MODEL_DIR) if os.path.isdir(MODEL_DIR) else []
        matches = [f for f in ggufs if fnmatch.fnmatch(f, model_match)]
        if len(matches) == 1:
            model = os.path.join("/models", matches[0])
        elif len(matches) > 1:
            print(f"  warning: multiple models match '{model_match}': {matches}")
            model = os.path.join("/models", matches[0])
        elif not ggufs:
            print("  warning: no GGUF files in llama/models/")

    # Start the container
    cmd = [
        "podman", "run", "-d", "--name", CONTAINER_NAME,
        "--network", "host",
        "--volume", f"{MODEL_DIR}:/models:ro",
        "--volume", f"{CONFIG_DIR}:/config:ro",
        "--volume", f"{data_dir}:/data:rw",
    ]

    image_mode = image_tag.split(":")[-1]

    if _has_cdi_nvidia():
        # CDI-based NVIDIA GPU passthrough (Podman-native, preferred)
        cmd.extend(["--device", "nvidia.com/gpu=all",
                     "--security-opt", "label=disable"])
    elif image_mode in ("rocm", "vulkan") and _has_rocm_gpu():
        # ROCm: kfd + dri, Vulkan: dri only (shared flags)
        if image_mode == "rocm":
            cmd.extend(["--device", "/dev/kfd",
                         "--device", "/dev/dri",
                         "--group-add", "video",
                         "--security-opt", "seccomp=unconfined"])
        else:
            cmd.extend(["--device", "/dev/dri",
                         "--group-add", "video"])

    # Build the llama-server command (override image entrypoint to pass model)
    server_args = ["-fa", "auto", "--jinja", "--metrics", "--slots",
            "--slot-save-path", "/data/slots",
            "--cors-origins", "localhost"]
    if model:
        server_args = ["-m", model] + server_args
    server_args.extend(["--ctx-size", "16384", "-ngl", "99", "-t", "9",
                 "-b", "4096", "-ub", "2048"])
    server_args.append("--api-key-file")
    server_args.append("/data/.api_key")

    cmd.append(image_tag)
    cmd.extend(server_args)

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  failed to start llama container: {result.stderr[:200]}")
        return False

    return True


def stop_llama_container():
    """Stop the llama container (called on app exit)."""
    try:
        subprocess.run(
            ["podman", "stop", CONTAINER_NAME],
            check=False, capture_output=True, timeout=10
        )
    except Exception:
        pass


def llama_container_running():
    """Check if the llama container is currently running."""
    result = subprocess.run(
        ["podman", "ps", "--filter", f"name={CONTAINER_NAME}",
         "--format", "{{.Status}}"],
        capture_output=True, text=True
    )
    return "Up" in result.stdout.strip()


def llama_health_url():
    """Return the health check URL for the llama container."""
    return "http://127.0.0.1:8080/health"
