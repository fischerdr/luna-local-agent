"""Backends: other places her model and voice can run, each in its own folder.

LM Studio, llama.cpp built here and kokoro-reader are the installer's
own. Anything else - containers, a server on the network - is
backends/<name>/, with a backend.py in it. Delete the folder and it's
gone; nothing else in Luna names it. (plugins/ is the other kind of
add-on: what she talks to while she runs. These are only seen by the
installer and the launcher, never by her.)

Every hook is a function in backend.py, and every one is optional.
backend.py is imported by the installer and the launcher, so it should
import only the standard library at the top and leave its real work
to the hooks.

Installer (installer/install.py). `inst` is the installer itself: its
ARGS, ROOT, SYSTEM and helpers (step, ok, warn, ask, choose, run, note,
set_config, load_state...).

    add_args(ap)               flags of its own, on the argparse parser
    validate(inst, args)       -> (errors, notes), before anything runs; may fill args in
    check_system(inst)         a line or two under "Checking this machine"
    backends(inst)             -> [{"key", "label", "menu", "default", "local_model"}]
                                  --backend <key> takes any of them; "menu" ones are
                                  offered when it asks, a "default" one is preselected,
                                  and "local_model": False skips picking a model here
                                  (called once before the flags are read, for --backend's
                                  choices, so inst.ARGS can still be None)
    backend(inst, state, key)  one of its backends was picked
    gpu_reserve(inst, state)   -> GB of graphics memory to keep beside her model
    model(inst, state, pick)   her model for its backend: a dict (key, label, repo, file,
                                  mmproj, ctx), or None when local_model is False
    voices(inst)               -> [{"key", "label", "default"}]: --voice <key> takes any;
                                  a "default" one is offered first when --voice isn't given
    voice(inst, state, key)    one of its voices was picked
    voice_failed(inst, state)  -> True if it set up her voice when kokoro-reader wouldn't install
    install(inst, state)       after her model is picked
    health(inst, state)        lines for the health check

Before any of that, installer/install.sh sources backends/<name>/python.sh,
if there is one, when it finds no Python 3.10-3.13: it may set PY to a
newer one, when its backend brings what the older Python was for.

Launcher (launch.py):

    start(settings)            before her: -> {"handled": {"chat", "tts"...}, "stop": fn},
                                  or None. What it handled isn't started again, and
                                  stop() runs when she exits.
"""
import importlib
import os
import sys

_found = None


def found():
    """Every backend here, imported once. One that won't import is said
    once and left out: an add-on mustn't be why Luna won't install or start."""
    global _found
    if _found is None:
        _found = []
        names = sorted({n for d in __path__ for n in os.listdir(d)
                        if n[0] not in "_." and os.path.isfile(os.path.join(d, n, "backend.py"))})
        for name in names:
            try:
                _found.append(importlib.import_module(f"{__name__}.{name}.backend"))
            except Exception as e:
                print(f"backends/{name} didn't load ({e}) - carrying on without it", file=sys.stderr)
    return _found


def call(hook, *args):
    """hook(*args) on every backend that has it: [(backend, result)]."""
    return [(b, getattr(b, hook)(*args)) for b in found() if callable(getattr(b, hook, None))]


def results(hook, *args):
    """What call() got back, without the Nones."""
    return [r for _b, r in call(hook, *args) if r is not None]


def offered(hook, *args):
    """What the backends() or voices() hooks offer: {key: (backend, entry)}."""
    return {e["key"]: (b, e) for b, entries in call(hook, *args) for e in entries or ()}
