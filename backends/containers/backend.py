"""The containers backend's hooks (backends/__init__.py says what each is for).

    --backend container   llama.cpp in rootless Podman containers here (Linux)
    --backend remote      her servers on another machine, running deploy/ there
    --voice container     the voice (Kokoro) in a container here

The work is in installer.py and launcher.py, imported only when a hook
runs; this file imports nothing that isn't the standard library.
"""
import platform
import shutil
import sys

LINUX = platform.system() == "Linux"
# kokoro and misaki 0.9.4 both say requires-python >=3.10,<3.13
KOKORO_PYTHON = ((3, 10), (3, 13))


def podman():
    return LINUX and bool(shutil.which("podman"))


def python_wants_containers(version=None):
    """A Python newer than install.sh picks (3.13), on Linux with Podman:
    the voice and the model server default to containers, which bring the
    Python and builds they need."""
    return tuple((version or sys.version_info)[:2]) > (3, 13) and podman()


def kokoro_python_ok(version=None):
    """Whether this Python can run Kokoro on the host."""
    return KOKORO_PYTHON[0] <= tuple((version or sys.version_info)[:2]) < KOKORO_PYTHON[1]


# --- installer ---------------------------------------------------------------
def add_args(ap):
    ap.add_argument("--remote", default=None, metavar="ADDR",
                    help="her servers on another machine (backends/containers/deploy/ there); "
                         "implies --backend remote")


def validate(inst, args):
    from backends.containers import installer

    return installer.validate(inst, args)


def check_system(inst):
    if python_wants_containers():
        print("    with Podman here, her voice and model server can run in containers instead")


def backends(inst):
    return [{"key": "container", "label": "llama.cpp in Podman containers - nothing to build",
             "menu": podman(), "default": python_wants_containers(), "local_model": True},
            {"key": "remote", "label": "on another machine - a GPU server on your network",
             "menu": True, "local_model": False}]


def backend(inst, state, key):
    if key == "container":
        inst.ok("set up after her model is picked")
    else:
        from backends.containers import installer

        installer.remote(inst, state)


def gpu_reserve(inst, state):
    from backends.containers import installer

    return installer.gpu_reserve(state)


def model(inst, state, pick):
    if pick is None:
        inst.ok("on the server - backends/containers/deploy/.env there says which")
        return
    from backends.containers import installer

    installer.container_model(inst, state, pick)


def voices(inst):
    """The container, offered first where the host voice would fail: on a
    Python Kokoro doesn't install in."""
    lo, hi = KOKORO_PYTHON
    why = "" if kokoro_python_ok() else \
        f" - Kokoro needs Python {lo[0]}.{lo[1]}-{hi[0]}.{hi[1] - 1}, this is {sys.version.split()[0]}"
    return [{"key": "container", "label": "in a Podman container (its own Python)" + why,
             "default": podman() and not kokoro_python_ok()}]


def voice(inst, state, key):
    from backends.containers import installer

    installer.voice_container(inst, state)


def voice_failed(inst, state):
    if podman():
        inst.warn("the voice server's packages didn't install here")
        if inst.ask("Run her voice in a Podman container instead (it brings its own Python)?"):
            from backends.containers import installer

            return installer.voice_container(inst, state)
    elif LINUX:
        print("  (with Podman installed, --voice container runs the voice in a container instead)")
    return False


def install(inst, state):
    if state.get("backend") == "container":
        from backends.containers import installer

        installer.containers(inst, state)


def health(inst, state):
    if state.get("backend") in ("container", "remote") or state.get("voice", {}).get("container"):
        from backends.containers import installer

        installer.health(inst, state)


# --- launcher ----------------------------------------------------------------
def start(settings):
    """config.json "backends": {"containers": {"autostart": true}}: her
    containers up (or kept) before her."""
    mine = (settings.get("backends") or {}).get("containers") or {}
    if mine.get("autostart") is not True or not LINUX:
        return None
    from backends.containers import launcher

    return launcher.start(mine)
