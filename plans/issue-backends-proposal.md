# Proposal: running her model and voice in containers, as an optional `backends/` add-on

This issue explains what I'd like to contribute, and why, before any pull request, so you can say what fits and what
doesn't. Nothing here changes Luna for a Mac with LM Studio. That's a design rule, not a side effect.

## Who this is for

You run Luna on macOS with LM Studio, and that path stays exactly as it is. Some of us run her differently:

- llama.cpp on Linux, on the same machine;
- llama.cpp on a GPU box elsewhere on the network, with Luna on a laptop or a small machine;
- a newer Linux where the host install of her voice doesn't work. Fedora 43 ships Python 3.14, and kokoro and misaki
  need Python <3.13.

## The problem

For us, getting Luna running means building llama.cpp on the host, and keeping a second Python around just for
Kokoro. On a small GPU, it also means working out by hand what fits beside the model. A Linux user with an 8 GB card
hits all three at once.

## The proposal

Rootless Podman containers for her four servers, as an add-on folder:

- chat (llama.cpp), on `:8080`;
- embeddings, on `:8081`;
- the reranker, on `:8082`;
- her voice (Kokoro), on `:8899`.

Luna herself doesn't change. She reaches the containers at the same addresses, with the same key (`llama/.api_key`),
as `llama/start.sh` and kokoro-reader. They're one more provider of the URLs she already uses, like LM Studio.

The same images also run on a Linux GPU server with compose, and the installer can point Luna at that server
(`--remote ADDR`). That works from a Mac too.

## Keeping the change small

I don't want this to spread through the codebase, so the proposal is in two layers.

**1. A small, generic hook layer in core: `backends/`.** A folder `backends/<name>/backend.py` can add choices to the
installer's `--backend` and `--voice`, set itself up, and start its servers from `launch.py` before her.

- The idea is the same as `plugins/`: drop the folder in and it's offered, delete it and it's gone.
- It's for the installer and the launcher only. Luna never loads a backend at run time.
- With no backend folder, the installer and launcher do exactly what they do today: the same choices, defaults and
  steps.
- The size: about 140 lines in `installer/install.py`, `installer/install.sh` and `launch.py`, plus an 84-line loader.

**2. The containers themselves, all in `backends/containers/`.** This adds files only and touches nothing outside its
folder.

- On macOS, `container` isn't offered (it's Linux only); `--remote` is.
- Without Podman, `container` isn't in the menu.

## Scope

**In:**

- the `backends/` loader and hooks, documented in `docs/development.md`, and tested with a pretend backend;
- containers on Linux:
  - images pinned by digest (CUDA, ROCm, Vulkan, CPU);
  - keys passed as Podman secrets;
  - everything on `127.0.0.1`;
  - `python -m backends.containers setup|up|down|status|logs|bench|reset`;
- a GPU server on the network: compose files for Podman or Docker, a firewall rule for one client (printed, never run),
  and `--remote` in the installer;
- placement on a small GPU: chat and voice first, the side models only if they fit, and the reranker off when a rerank
  on the CPU is too slow to finish in time;
- `config.json` `"backends": {"containers": {"autostart": true}}`, so `./start.sh` starts and stops them with her;
- tests: unit tests (no Podman needed), live container tests (opt-in), and a from-zero reset and check script.

**Out:**

- any change to how Luna works at run time: `main.py`, memory, tools, prompts and `plugins/` stay as they are;
- LM Studio, the host llama.cpp build and the host kokoro-reader: unchanged, and still the default;
- Windows and macOS containers (Podman there runs a Linux VM, with no GPU passthrough worth having);
- starting the embedder and reranker on demand: it would need a hook in `embedmem.py`, so it's left for later, if
  ever;
- the wake word on Linux: an upstream openwakeword packaging problem, as a separate issue.

## How it would arrive

| Order | PR | Size | Depends on |
|---|---|---|---|
| any time | three small fixes found along the way (below) | 1 line each, plus tests | nothing |
| 1 | `backends/`: the hooks only, no backend | ~140 lines in core | nothing |
| 2 | `backends/containers/` | new folder only | PR 1 |

PR 2 would be open as a draft beside PR 1, so the hooks are seen next to what uses them.

The fixes don't depend on any of this:

1. **Speech input fails on every fresh install.** PyAV 19 removed an argument faster-whisper passes. The fix pins
   `av<19`.
2. **`/settings` crashes.** It's a stray `import config` inside `handle_input()`.
3. **The Live2D runtime isn't gitignored** in the folder where the installer saves it.

## What I'd like your view on

1. **The approach.** Is an optional add-on folder for other model and voice servers something you'd take? Or would you
   rather this lived outside the repo?
2. **The name.** Is `backends/` OK beside `plugins/`? It means where her model and voice run, as opposed to what she
   talks to.
3. **The hooks.** The list is in the `docs/development.md` section of PR 1, 13 optional functions. Too many, too few,
   named differently?
4. **The config key.** `"backends": {"containers": {...}}` copies the shape of `"plugins"`. Is that OK?
5. **Upkeep.** I'd look after `backends/containers/`: image pins, llama.cpp updates, breakages. Does that work for you,
   or would you rather not carry it at all?
6. **Testing.** Its unit tests run separately
   (`python -m unittest discover -s backends/containers/tests -t .`). Should they join the main test run?

It's all built and tested on Fedora 43 with an RTX 3070 (8 GB). From zero, chat and voice go on the GPU (6.3 of 8 GB),
the embedder on the CPU, and the reranker off. The branches are on my fork, `fischerdr/luna-local-agent`, if you want
to look before any PR:

- `feat/plugin-hooks`
- `feat/containers-plugin`
- `fix/av-19`
- `fix/settings-config`
- `fix/gitignore-live2d`
