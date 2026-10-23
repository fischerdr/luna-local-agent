"""The containers with Luna: launch.py's start hook (backend.py start()).

config.json "backends": {"containers": {"autostart": true}} brings them
up (or keeps them) before her, the way python -m backends.containers up
does, and down again when she exits unless "stop_on_exit" is false.
Containers that were already running are left alone.
"""
import shutil
import sys
import time


def start(mine):
    """Her containers up before her. Never holds her start up: a role that
    won't come gets one line. {"handled": roles, "stop": fn} or None."""
    if not shutil.which("podman"):
        print("backends.containers.autostart is on, but Podman isn't installed - starting without the containers.",
              file=sys.stderr)
        return None
    try:
        from backends.containers import keys, lifecycle, settings

        s = settings.load()
        results = lifecycle.up(s, lifecycle.default_roles(s), keys.token_source()[0])
    except Exception as e:
        print(f"The containers didn't come up ({e}) - starting without them.", file=sys.stderr)
        return None

    handled = {r.role: r.state for r in results if r.state in ("kept", "started")}
    for r in results:
        if r.role not in handled:
            print(f"  {r.role} container: {r.note}", file=sys.stderr)
    timeout = float(s.get("READY_TIMEOUT") or 600)
    try:
        for role in handled:
            port = settings.port(role, s)
            print(f"Starting her {role} container", end="", flush=True)
            ready = lifecycle.wait_ready(f"http://127.0.0.1:{port}", timeout, alive=lifecycle.running(role),
                                         pause=lambda sec: (time.sleep(sec), print(".", end="", flush=True)))
            print(" ready." if ready else " still not ready - python -m backends.containers logs " + role)
    except KeyboardInterrupt:
        print(" - not waiting; it carries on loading")

    started = [role for role, how in handled.items() if how == "started"]

    def stop():
        if mine.get("stop_on_exit", True) is False or not started:
            return
        try:
            lifecycle.down(started)
        except Exception as e:
            print(f"Couldn't stop the containers ({e}) - python -m backends.containers down does.", file=sys.stderr)

    return {"handled": set(handled), "stop": stop}
