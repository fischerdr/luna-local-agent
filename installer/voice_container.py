"""Voice (kokoro) container management via Podman.

Called by launch.py at startup and install.py at install time.
The kokoro_reader server runs unmodified inside the container.

Supports:
  - CPU mode (python:3.12-slim, CPU PyTorch)
  - CUDA mode (nvidia/cuda runtime, GPU-accelerated)

Remote expansion hooks (future):
  - Remote TTS: change tts.url to remote host (config-only)
  - Remote LLM: change LLM_URL env var (config-only)
"""
import json
import os
import shutil
import subprocess

from .install import warn, fail, ok, step, note, http_ok, venv_python

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IMAGE_NAME = "luna-tts"
CONTAINER_NAME = "luna-tts"
DEFAULT_TTS_URL = "http://127.0.0.1:8899"


def voice_container_mode():
    """Read voice mode from install.json.
    
    Returns: "container", "container_cuda", "managed", "external", "none", etc.
    """
    try:
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "install.json")) as f:
            data = json.load(f)
        return data.get("voice", {}).get("mode", "container_cuda")
    except (FileNotFoundError, KeyError):
        return "container_cuda"


def tts_url_from_config():
    """Return the TTS server URL from config or install.json."""
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "config.json")) as f:
            cfg = json.load(f)
        url = (cfg.get("tts") or {}).get("url")
        if url:
            return str(url).rstrip("/")
    except (OSError, ValueError):
        pass
    return DEFAULT_TTS_URL


def _dockerfile_name(mode):
    """Map mode string to Dockerfile name."""
    if mode == "container_cuda":
        return "Dockerfile.cuda"
    return "Dockerfile"  # CPU mode


def build_voice_container(mode=None):
    """Build the TTS container image from kokoro-reader source.
    
    Args:
        mode: "container" (CPU) or "container_cuda" (NVIDIA GPU).
              If None, detected from system capabilities.
    
    Returns:
        True on success, False on failure.
    """
    import tempfile
    
    if mode is None:
        # Auto-detect: prefer CUDA if NVIDIA GPU + container toolkit available
        if _has_nvidia_gpu():
            mode = "container_cuda"
        else:
            mode = "container"
    
    if mode == "container_cuda":
        if not shutil.which("nvidia-container-toolkit"):
            warn("nvidia-container-toolkit not found — falling back to CPU mode")
            mode = "container"
    
    image_tag = f"{IMAGE_NAME}:{mode}"
    dockerfile = _dockerfile_name(mode)
    
    step(f"Building voice container: {image_tag}")
    
    tmpdir = tempfile.mkdtemp(prefix="luna-tts-build-")
    try:
        # Clone kokoro-reader if needed
        kokoro_dir = os.environ.get(
            "LUNA_KOKORO_DIR",
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         os.pardir, "kokoro-reader")
        )
        
        server_dir = os.path.join(kokoro_dir, "server")
        has_server = os.path.isfile(os.path.join(server_dir, "kokoro_server.py"))
        
        if not has_server:
            import subprocess
            from .install import KOKORO_REPO
            print(f"  cloning kokoro-reader to {kokoro_dir}")
            subprocess.run(
                ["git", "clone", "--depth", "1", KOKORO_REPO, kokoro_dir],
                check=True, capture_output=True
            )
        
        # Copy server files into build context
        for f in ["kokoro_server.py", "blends.json", "start_server.sh"]:
            src = os.path.join(server_dir, f)
            if os.path.isfile(src):
                shutil.copy2(src, tmpdir)
        
        # Write Dockerfile to build context
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dockerfile_src = os.path.join(repo_root, "tts", dockerfile)
        if os.path.isfile(dockerfile_src):
            shutil.copy2(dockerfile_src, os.path.join(tmpdir, "Dockerfile"))
        else:
            # Fallback: embed the Dockerfile inline
            _write_inline_dockerfile(tmpdir, mode)
        
        # Build image
        result = subprocess.run(
            ["podman", "build", "-t", image_tag, tmpdir],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            fail(f"Failed to build voice container")
            print(f"  {result.stderr[:500]}")
            return False
        
        ok(f"container image built: {image_tag}")
        return True
    
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _write_inline_dockerfile(tmpdir, mode):
    """Write Dockerfile inline (fallback when tts/Dockerfile.* not found)."""
    if mode == "container_cuda":
        content = """FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04
RUN apt-get update && apt-get install -y --no-install-recommends espeak-ng python3-pip && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch ctranslate2 kokoro>=0.9.4 numpy>=1.24 soundfile
WORKDIR /opt/kokoro-reader/server
ENV KOKORO_HOST=0.0.0.0 KOKORO_PORT=8899 KOKORO_LANG=a KOKORO_DEVICE=cuda KOKORO_VOICE=am_adam
EXPOSE 8899
COPY kokoro_server.py blends.json start_server.sh ./
CMD ["python", "kokoro_server.py"]
"""
    else:
        content = """FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends espeak-ng && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "torch --index-url https://download.pytorch.org/whl/cpu" ctranslate2 kokoro>=0.9.4 numpy>=1.24 soundfile
WORKDIR /opt/kokoro-reader/server
ENV KOKORO_HOST=0.0.0.0 KOKORO_PORT=8899 KOKORO_LANG=a KOKORO_DEVICE= KOKORO_VOICE=am_adam
EXPOSE 8899
COPY kokoro_server.py blends.json start_server.sh ./
CMD ["python", "kokoro_server.py"]
"""
    with open(os.path.join(tmpdir, "Dockerfile"), "w") as f:
        f.write(content)


def _has_nvidia_gpu():
    """Check if an NVIDIA GPU is available on the host."""
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


# ---------------------------------------------------------------------------
# Container lifecycle
# ---------------------------------------------------------------------------

def ensure_voice_container(mode=None):
    """Start the voice container if not already running. Idempotent.
    
    Args:
        mode: "container" or "container_cuda". If None, reads install.json.
    
    Returns:
        True if container is running, False on failure.
    """
    name = CONTAINER_NAME
    if mode is None:
        mode = voice_container_mode()
    image_tag = f"{IMAGE_NAME}:{mode}"
    
    # Check if container exists and its status
    result = subprocess.run(
        ["podman", "ps", "-a", "--filter", f"name={name}",
         "--format", "{{.Status}}"],
        capture_output=True, text=True
    )
    status = result.stdout.strip()
    
    if not status:
        # Container doesn't exist — build it first
        print("  voice container not found. Building...")
        if not build_voice_container(mode):
            warn("Container build failed — will fall back later")
            return False
        # Build created the image — start container below
    
    elif "Exited" in status or "exited" in status:
        # Container exists but stopped — restart it
        subprocess.run(
            ["podman", "start", name],
            check=True, capture_output=True
        )
        ok("container restarted")
    elif "Up" in status:
        # Already running — nothing to do
        return True
    
    # Start the container (new or restarted)
    cmd = ["podman", "run", "-d", "--name", name, "--network", "host"]
    
    if mode == "container_cuda":
        cmd.extend(["--gpus", "all"])
    
    cmd.append(image_tag)
    
    result = subprocess.run(
        cmd, capture_output=True, text=True
    )
    if result.returncode != 0:
        fail(f"Failed to start voice container: {result.stderr}")
        return False
    
    ok("voice container started")
    return True


def stop_voice_container():
    """Stop the voice container (optional, called on app exit)."""
    try:
        subprocess.run(
            ["podman", "stop", CONTAINER_NAME],
            check=False, capture_output=True, timeout=10
        )
    except Exception:
        pass  # Non-fatal on shutdown


def health_check(url=None):
    """Check if the voice container is responding.
    
    Same health check the main app uses.
    """
    url = url or tts_url_from_config()
    try:
        return http_ok(url, timeout=3)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Remote expansion helpers (future)
# ---------------------------------------------------------------------------

def remote_tts_tunnel(remote_host):
    """Create an SSH tunnel to a remote TTS server.
    
    Call this before tts_up() to route local :8899 to a remote host.
    Example: remote_tts_tunnel("user@remote-host")
    
    Returns:
        SSH subprocess handle (keep alive for tunnel to persist).
    """
    return subprocess.Popen([
        "ssh", "-L", "8899:127.0.0.1:8899", "-N", remote_host
    ])


def remote_llm_tunnel(remote_host):
    """Create an SSH tunnel to a remote LLM server.
    
    Call this before launch.py starts the app.
    Example: remote_llm_tunnel("user@remote-host")
    
    After tunneling, set LLM_URL=http://127.0.0.1:8080/v1/chat.
    """
    return subprocess.Popen([
        "ssh", "-L", "8080:127.0.0.1:8080", "-N", remote_host
    ])
