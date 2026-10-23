# backends/: a folder for other places her model and voice can run

Branch: `feat/plugin-hooks`, from `main` (`f27ef43`).

## What this does

The installer knows LM Studio, llama.cpp built here and kokoro-reader. This PR lets a folder add more choices without
editing the installer each time. A folder `backends/<name>/` with a `backend.py` can:

- add choices to `--backend` and `--voice`;
- add to the model-server and voice questions;
- set itself up;
- start its servers before her (from `launch.py`).

Drop the folder in and the installer offers it. Delete it and it's gone; nothing else names it.

It works like `plugins/`, but for the installer and the launcher only. Luna herself never loads a backend; she talks to
the URLs it set up, as she does with LM Studio's.

**This PR adds no backend.** With no `backends/<name>/` folder, the installer and the launcher do exactly what they do
on main:

- the same `--backend` choices (`lmstudio`, `llama`, `skip`);
- the same questions, defaults and steps;
- the same `./start.sh` flow.

On a Mac with LM Studio, nothing changes.

The first backend to use this is containers (llama.cpp and Kokoro in Podman, or on a GPU server on the network), in a
follow-up PR that adds only `backends/containers/`.

## Changes

| File | Change |
|---|---|
| `backends/__init__.py` | new: the loader and the hook list. One that fails to import is reported and skipped. |
| `installer/install.py` | +89/-19: a hook call at each point below; `parser()` split out of `main()`; `pick_model(vram, reserve=0)` |
| `installer/install.sh` | +5: sources `backends/<name>/python.sh` when no Python 3.10-3.13 is found |
| `launch.py` | +39: `start_backends()` before her, its `stop()` after; doesn't start what a backend handled |
| `docs/development.md` | a "Backends" section with the hook table |
| `docs/install.md` | one paragraph pointing at it |
| `tests/test_backends.py` | new, 15 tests |

## The hooks

Every hook is optional.

| Hook | Called | For |
|---|---|---|
| `add_args(ap)` | building the flags | flags of its own |
| `validate(inst, args)` → `(errors, notes)` | before anything runs | its flags against the others |
| `check_system(inst)` | "Checking this machine" | a line about what it needs |
| `backends(inst)` → `[{key, label, menu, default, local_model}]` | `--backend`, and the model-server question | the model servers it offers |
| `backend(inst, state, key)` | one of them is picked | setting it up |
| `gpu_reserve(inst, state)` → GB | picking her model | graphics memory kept for what shares the GPU |
| `model(inst, state, pick)` | after the model pick | where its backend gets the model |
| `voices(inst)` → `[{key, label, default}]` | `--voice`, and the voice question | the voices it offers |
| `voice(inst, state, key)` | one of them is picked | setting it up |
| `voice_failed(inst, state)` → bool | kokoro-reader's packages didn't install | offering its voice instead |
| `install(inst, state)` | after her model | the rest of its setup |
| `health(inst, state)` | the health check | its own lines |
| `start(settings)` → `{handled, stop}` | `launch.py`, before her | starting its servers |

`inst` is the installer: `inst.ARGS` and its helpers (`ok`, `warn`, `ask`, `choose`, `run`, `note`, `set_config`...).

A backend whose `start()` raises is reported and skipped, so it never stops her from starting.

## Two small behaviour notes

- `--voice` now defaults to `None` instead of `"install"`, so a backend can tell "not given" from "install".
  `voice()` treats `None` as `install`, so the result is the same.
- `pick_model(vram, reserve=0)`: a backend can keep some graphics memory for what shares the GPU (a voice on the same
  card). With no backend, `reserve` is 0 and the pick is main's.

## Testing

```bash
python -m unittest discover tests      # 358 tests, OK (2 skipped)
python installer/install.py --help     # --backend {lmstudio,llama,skip}, as on main
python installer/install.py --check
./start.sh --check
```

`tests/test_backends.py`:

- with no backend: the same choices, and no extra launcher steps;
- a pretend backend that uses every hook (menu, defaults, the model pick with memory kept for it, its voice, the
  launcher skipping what it handled);
- a broken backend is skipped with a message;
- `install.sh` with a too-new Python: nothing happens without a `python.sh`, and a `python.sh` picks the Python.

Tested on Fedora 43 (Linux). I haven't run it on macOS or Windows: the change there is one empty loop at each hook,
but a run on a Mac would be welcome.
