# backends/containers: her model and voice in Podman containers, here or on a GPU server

Branch: `feat/containers-plugin`. Builds on `feat/plugin-hooks` (the `backends/` PR): merge that one first.

## What this does

It runs her model, the embedder, the reranker and her voice in rootless Podman containers, so nothing gets built or
installed on the host. Or it runs all four on a Linux GPU server on the network.

Luna doesn't change. She reaches them at the same addresses (`:8080`, `:8081`, `:8082`, `:8899`) with the same key
(`llama/.api_key`) as `llama/start.sh` and kokoro-reader.

```bash
./installer/install.sh --backend container --voice container   # here (Linux, Podman)
./installer/install.sh --remote 192.168.1.50                   # a GPU server running backends/containers/deploy/
```

**Nothing outside `backends/containers/` changes.** Everything plugs in through the `backends/` hooks.

- **On macOS:** the installer doesn't offer `container` (it's Linux only). `--remote` is available, for a Mac pointing
  at a Linux GPU server.
- **Without Podman:** `container` isn't in the menu.
- **To remove it:** delete the folder, and Luna installs and starts exactly as on main.

## Why

- **No build on the host.** llama.cpp comes from upstream's images (`ghcr.io/ggml-org/llama.cpp`, pinned by digest),
  with CUDA, ROCm, Vulkan and CPU variants.
- **No voice Python on the host.** kokoro and misaki need Python <3.13. Fedora 43 ships 3.14, where the host voice
  can't install at all. The voice image brings its own Python 3.12.
- **A GPU server.** People with the GPU in another machine can run all four there with compose, and point Luna at it.
- **Fitting a small GPU.** On an 8 GB card, chat and the voice go on the GPU first. The embedder and reranker go there
  only if there's room; otherwise they use the CPU. The reranker is turned off if a rerank on the CPU is too slow to
  finish in time. `python -m backends.containers bench` measures this and writes it down.

## What's in the folder

| Path | What |
|---|---|
| `backend.py` | the hooks (standard library only). It offers `--backend container`, `--backend remote` and `--voice container`, and adds `--remote ADDR`. |
| `installer.py`, `launcher.py` | what the hooks do, imported only when one runs |
| `__main__.py`, `setup.py`, `lifecycle.py`, `run.py`, `settings.py`, `images.py`, `keys.py`, `bench.py` | `python -m backends.containers setup\|up\|down\|status\|logs\|bench\|reset` |
| `llama/`, `tts/` | the two images (a thin layer over upstream llama.cpp; Kokoro on `python:3.12-slim`) |
| `deploy/` | compose files for a GPU server (Podman or Docker; CUDA, ROCm, Vulkan) |
| `services.env` | its defaults. This machine's own go in `services.local.env` (gitignored). |
| `python.sh` | lets `install.sh` go on with Python 3.14+ when Podman is here |
| `README.md`, `docs.md` | what it is, and the full reference |
| `tests/` | unit tests, live container tests, and `tests/zero/` (a from-zero reset and check) |

## How it fits Luna

- Settings are read by bash in `llama/start.sh`'s order: `llama/server.env`, then `server.local.env`, then
  `services.env`, then `services.local.env`. Her `server.env` switches (`EMBED`, `RERANK`, `CTX`, `NGL`...) keep their
  meaning.
- The API key is `llama/.api_key`, as on main. It goes into the containers as a Podman secret.
- With `config.json` `"backends": {"containers": {"autostart": true}}`, `./start.sh` brings the containers up before
  her, and stops the ones it started when she exits. While her model is in a container, `llama/start.sh` isn't
  started; while her voice is, kokoro-reader isn't.
- `--remote` writes her three URLs into `config.json` and probes each service with the key.
- Containers and ports are on `127.0.0.1` only. A GPU server binds to its own address, and gets a firewall rule for
  one client (printed, never run).
- `reset` only removes what it made: `luna-*` containers, volumes, secrets and images. It asks first, and asks again
  for downloaded models.

## Testing

```bash
python -m unittest discover tests                              # core: 358 OK
python -m unittest discover -s backends/containers/tests -t .  # this folder: 291 OK (52 live ones skipped)
LUNA_CONTAINER_TESTS=1 python -m unittest discover -v -s backends/containers/tests/live -t .   # with Podman
backends/containers/tests/zero/reset.sh --install                # from zero: reset, install, check (asks first)
backends/containers/tests/zero/check.sh                          # check a running setup
```

Tested on Fedora 43 with an RTX 3070 (8 GB), CUDA 13, rootless Podman 5, and the uncensored 4B. From zero, chat and
voice end up on the GPU (6.3 of 8 GB), the embedder on the CPU, and the reranker off. `check.sh` passes everything:

- chat answers, and refuses without the key;
- embeddings work, and the voice speaks;
- Luna's own code recalls by meaning;
- the ports listen on 127.0.0.1 only.

The compose files are checked against the Podman path by `tests/test_compose_drift.py`. Docker Compose v2 was
validated through the `docker:cli` image.

## Not in this PR

- **Embedder and reranker on demand** (started when a turn wants them, stopped when idle). It needs a hook in
  `embedmem.py`, which is Luna's runtime, so it's left for a later PR if wanted.
- **Wake word on Linux.** It's broken by an upstream packaging issue in openwakeword, on its own branch later.
