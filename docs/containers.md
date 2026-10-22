# Containers

Luna's backend can run in rootless Podman containers instead of on the
host: her model (llama.cpp's server), the embedding model, the reranker
and her voice (Kokoro). Or on a GPU server elsewhere on your network,
with Docker or Podman Compose.

Luna herself doesn't change. She reaches each server at the same address
with the same key (`llama/.api_key`) as she would llama/start.sh and the
installer's voice. Nothing is built on the host and no Python packages
are installed for these servers.

Linux only. On Windows or macOS, use LM Studio, or put the containers on
a Linux machine and point Luna at it ([another machine](#a-gpu-server-on-your-network)).

## What runs where

| Container | Port | Image | What it does |
|---|---|---|---|
| `luna-llama` | 8080 | `localhost/luna-services-llama:<gpu>` | her model, llama.cpp's server |
| `luna-embed` | 8081 | `localhost/luna-services-llama:cpu` | embeddings, for recall by meaning |
| `luna-rerank` | 8082 | `localhost/luna-services-llama:<gpu or cpu>` | the reranker |
| `luna-tts` | 8899 | `localhost/luna-services-tts:<cpu or cuda>` | her voice, kokoro-reader |

The ports are llama/server.env's `PORT`, `EMBED_PORT` and `RERANK_PORT`.
They're published on `HOST` from that file, which is `127.0.0.1`: this
machine only. The voice is always on `127.0.0.1:8899`, because it takes
no key.

The models live in the `luna-models` volume, saved chat slots in
`luna-data`, and the voice's model in `luna-tts-cache`. The key, and your
Hugging Face token if you have one, are Podman secrets.

The introspection lab (`introspection/`) isn't in a container. It needs
llama.cpp built here (`llama/install.sh`); see [Her reasoning](reasoning.md).

## What you need

- Podman 4.7 or newer, rootless (as yourself, not root). Fedora:
  `sudo dnf install podman`. Ubuntu and Debian: `sudo apt install podman`.
- For an NVIDIA GPU: nvidia-container-toolkit and its CDI spec (see
  [GPUs](#gpus)). AMD: `/dev/kfd` for ROCm, or `/dev/dri` for Vulkan.
- Optional: a Hugging Face token, for gated models. Public ones don't need it.

Luna's own Python can be newer than the host install allows (3.14, say):
her voice (Kokoro needs 3.10-3.12) and llama.cpp then run in their
containers. `install.sh` goes on with such a Python when Podman is there,
and offers containers for both.

## Setting it up

From the installer:

```bash
./installer/install.sh --backend container --voice container
```

It picks a model for your GPU, as it does for llama.cpp, and hands over to
`python -m services setup`. You can also run that yourself, in Luna's
Python (`.venv/bin/python -m services setup`):

```bash
python -m services setup          # the model servers
python -m services setup tts      # her voice
```

Setup asks before each step and shows the command first: building the
images, the volumes, the secrets, which model to use. Run it again any
time; it does only what's missing. At the end it offers to start the
containers with Luna. Then:

```bash
luna        # or ./start.sh
```

## Commands

```bash
python -m services setup              # set up the model servers
python -m services setup tts          # set up her voice
python -m services up [ROLE...]       # start them, or keep them if they're current
python -m services down [ROLE...]     # stop and remove them (models and secrets stay)
python -m services status             # each one, and whether it answers
python -m services logs ROLE [-f]     # a container's log
python -m services bench              # where the side models should run on this GPU
python -m services reset              # remove what it made (asks first)
```

Roles are `chat`, `embed`, `rerank` and `tts`. `up` with no roles starts
the three llama roles, and the voice once `setup tts` has moved it into
its container.

`up` is safe to repeat. Each container carries a hash of everything that
makes it up: its settings, image, model, key and token. One that matches
is left running. When anything changes, it's made anew. Up waits for
each to answer (Ctrl+C stops the wait, not the container). It won't take
a port something else is using: a host llama/start.sh or LM Studio is
named and left alone.

`reset` asks before removing anything, then asks separately about
`luna-models` (your downloads), and about `luna-*` containers or images
something else made.

## Starting with Luna

In config.json:

```json
"services": {"autostart": true, "stop_on_exit": true}
```

With `autostart`, `./start.sh` runs `python -m services up` before her,
and waits for each container with dots. If one won't come, she starts
anyway and says why in one line. A chat container, even one still
loading, means llama/start.sh isn't started beside it. A voice container
means the host's kokoro-reader isn't started either. When she exits, the
containers it started are stopped again, unless `stop_on_exit` is false.
Ones that were already running are left alone.

Without a `services` block, `./start.sh` does exactly what it always did.

## Models

Each model is settled when the container is made, never searched for
while it runs. For each role, the first of these that's set wins:

1. `MODEL_HF` (or `EMBED_HF`, `RERANK_HF`): a Hugging Face model, as
   `<user>/<repo>[:quant]`. It downloads once into `luna-models`.
2. `MODEL` (or `EMBED_MODEL`, `RERANK_MODEL`): a file. One under
   `LUNA_MODELS_DIR` is used where it is, subfolders and all. A path
   starting `/models/` is a file already in the volume.
3. `MODEL_MATCH` (or `EMBED_MATCH`, `RERANK_MATCH`): llama/start.sh's
   search, in `LUNA_MODELS_DIR` and then the volume.

`LUNA_MODELS_DIR` is a folder of your own GGUFs, mounted read-only:
`llama/models`, or LM Studio's models folder. They're used without a
second copy. Setup offers the ones it finds. The embedding and reranker
files already there are offered instead of downloading them again.

If nothing is found for chat, `up` says how to set one. The embedder and
reranker are skipped when their switch is `auto`, as llama/start.sh does,
and refuse to start when it's `on`.

Vision follows `MMPROJ` as in [Vision](vision.md): `off` turns it off; a
file next to the model is used; an `-hf` model brings its repo's own
mmproj. `MMPROJ_HF="<user>/<repo>/<file>.gguf"` takes one from another
repo.

The model's name in her header is the file's name, or the repo's for an
`-hf` model, unless `ALIAS` says otherwise.

## Settings

The containers read the same files as llama/start.sh, in the same order,
and then two of their own. Each later one wins:

1. `llama/server.env` (or the file `LLAMA_ENV` names instead)
2. `llama/server.local.env`
3. `services/services.env`
4. `services/services.local.env`: yours, not committed

server.env's switches stay the switches: `EMBED`, `RERANK` and `MMPROJ`,
and `CTX`, `NGL`, `PARALLEL`, `KV_UNIFIED`, `CTX_CHECKPOINTS`, `THREADS`,
`BATCH`, `UBATCH`, `REASONING_FORMAT`, `REASONING_BUDGET`, `ALIAS`,
`EXTRA`, the ports and `HOST`.

services.env adds:

| Setting | Default | What it does |
|---|---|---|
| `MODEL_HF` | (none) | her model from Hugging Face |
| `MMPROJ_HF` | (none) | a vision adapter from another repo |
| `EMBED_HF` | `Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0` | the embedding model |
| `RERANK_HF` | `ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF:Q8_0` | the reranker |
| `LUNA_MODELS_DIR` | (none) | your own GGUFs, mounted read-only |
| `EMBED_NGL` | `0` | the embedder's GPU layers: 0 is the CPU |
| `RERANK_NGL` | `99` | the reranker's GPU layers |
| `LLAMA_BACKEND` | `auto` | the image: `auto`, `cpu`, `cuda`, `cuda13`, `rocm`, `vulkan` |
| `TTS_DEVICE` | `auto` | the voice: `auto` (the GPU on NVIDIA), `cpu`, `cuda` |
| `READY_TIMEOUT` | `600` | seconds `up` waits for each to answer |
| `EMBED_ONDEMAND` | `off` | `on`: the embedder and reranker start when wanted (below) |
| `EMBED_IDLE`, `RERANK_IDLE` | `600` | seconds without use before an on-demand one is stopped |

A role with no GPU layers (`EMBED_NGL=0`) runs in the cpu image, with no
GPU at all, so it takes no video memory.

## Small GPUs

On an 8 GB card, a 4B chat model, the embedder and the reranker fit
together; a bigger model or a long `CTX` may not. The embedder runs on
the CPU anyway: a query is a few dozen tokens, about 30 ms there.

```bash
python -m services bench
```

measures where the reranker should go. It goes on the GPU when chat and
it fit with 10% to spare. Otherwise it goes on the CPU, if a rerank of 24
facts takes under 1.5 seconds there. Otherwise it's turned off
(`RERANK=off`), and recall keeps the embedding order. It writes the
answer only if you say so: `EMBED_NGL` and `RERANK_NGL` to
services.local.env, `RERANK=off` to llama/server.local.env. Setup offers
to run it once.

Or have them only while they're used: `EMBED_ONDEMAND=on` in
services.local.env. Then `up` and `./start.sh` leave the embedder and
reranker stopped. The first time a turn wants one, it's started in the
background. That turn doesn't wait: it recalls by matching words, as it
would without one, and the next turn finds the server. When it hasn't
been wanted for `EMBED_IDLE` (or `RERANK_IDLE`) seconds, it's stopped
again. Only ones started this way are stopped, and only when they run
on this machine.

The voice on the GPU wants about 1 GB too. On a small card,
`TTS_DEVICE=cpu` leaves that to chat; Kokoro speaks faster than real time
on the CPU.

## GPUs

The image is picked from what's in the machine:

1. NVIDIA, with a CDI spec: `cuda13` when the driver's CUDA (nvidia-smi's
   header) is 13 or newer, else `cuda`.
2. NVIDIA without one: `cpu`, and setup prints the command that makes it:
   `nvidia-ctk cdi generate --output=$HOME/.config/cdi/nvidia.yaml`, or
   with root, `--output=/etc/cdi/nvidia.yaml`.
3. AMD with `/dev/kfd`: `rocm` (on ARM, `vulkan`).
4. `/dev/dri`: `vulkan`.
5. Otherwise `cpu`.

`LLAMA_BACKEND`, or the installer's `--gpu`, overrides it. The voice uses
CUDA only on NVIDIA with CDI. An AMD machine's voice runs on its CPU.

## A GPU server on your network

The four servers can run on another machine (a Linux box with a big GPU)
with Docker or Podman Compose. Luna's machine runs none of them. There
are no SSH tunnels; it's plain HTTP on your LAN.

On the server, from a clone of this repo:

```bash
cd deploy
cp .env.example .env          # set LUNA_BIND (its own address) and MODEL_HF
python -m services setup --server --allow 192.168.1.20    # the client's address
podman compose -f compose.yaml -f compose.cuda.yaml up -d
```

`setup --server` makes `deploy/api_key` and an empty `deploy/hf_token`
(empty means no token). It then **prints** a firewall rule that lets
only that one client in, for firewalld, ufw and nftables. It never runs
them. It also says which compose files fit the server's GPU:

| GPU | Podman | Docker |
|---|---|---|
| none | `compose.yaml` | `compose.yaml` |
| NVIDIA | `+ compose.cuda.yaml` (CDI) | `+ compose.cuda.docker.yaml` (needs nvidia-container-toolkit) |
| AMD (ROCm) | `+ compose.rocm.yaml` | `+ compose.rocm.yaml` |
| AMD or Intel (Vulkan) | `+ compose.vulkan.yaml` | `+ compose.vulkan.yaml` |

Compose won't start without `LUNA_BIND`, so the ports are never open on
every address by accident.

On Luna's machine:

```bash
./installer/install.sh --remote 192.168.1.50
```

The installer sets her URLs to the server and turns `llm.autostart` off,
so a llama.cpp build here never starts uselessly beside it. It asks for
the server's `api_key` (paste it, or give its path), and saves it as
`llama/.api_key`, readable only by you. Then it checks all four servers
and says what's wrong: the wrong key, or no answer (is the server up,
is `LUNA_BIND` its own address, does its firewall let this machine in?).

**This is plain HTTP.** The key guards her model, the embedder and the
reranker, but it crosses your network unencrypted. The voice takes no
key at all. The firewall rule that admits only your machine is the
second layer. Use it on a network you trust.

## Security

- Rootless Podman is the boundary: the servers run as root inside their
  containers, which is you outside.
- No `--network host`. Each port is published on `HOST` (127.0.0.1 unless
  you share it), and the voice only on 127.0.0.1.
- The key and token are Podman secrets, files at `/run/secrets/api_key`
  and `/run/secrets/hf_token` inside. They're never in a command line, an
  image, an env-file, `podman inspect` or a log.
- The images are built from `services/llama` and `services/tts` alone, so
  nothing from the repo (config.json, keys, models) can end up in one.
  Their bases are pinned by digest.
- `LUNA_MODELS_DIR` is mounted read-only and never relabelled. On SELinux
  the container is let through instead (`label=disable`).
- `HOST` other than 127.0.0.1 means other machines can reach them; `up`
  says so and asks.

## Feature parity

Everything that reaches a server works as it does with llama/start.sh:

| Feature | How |
|---|---|
| Token confidence, thinking budgets, second look | the same llama.cpp server, with `--reasoning-format` and `--reasoning-budget` from server.env |
| Vision | the `MMPROJ` rules above |
| Tool calling | `--jinja` |
| The model's name | `ALIAS`, or the file's (or repo's) name |
| The key and CORS | `--api-key-file`, `--cors-origins localhost` |
| Saved slots | `/data/slots`, in `luna-data` |
| Short on memory | `EMBED`, `RERANK`, `MMPROJ`, `CTX`, `PARALLEL` |
| Recall, knowledge, web, tools | the embedder (`--pooling last`) and reranker (`--pooling rank`) on :8081 and :8082 |
| Her voice | kokoro-reader as it is: `POST /tts`, `GET /health`, `GET /voices` |

## Troubleshooting

- **What's it doing?** `python -m services logs chat` (or `podman logs luna-llama`).
- **401 Unauthorized:** the key doesn't match. Luna reads `llama/.api_key`;
  the containers have the secret made from it. Run `python -m services up`
  after changing it. With a remote server, its `deploy/api_key`.
- **No GPU in the container:** check the CDI spec (`nvidia-ctk cdi list`),
  then `python -m services setup` again.
- **Port taken:** a host llama/start.sh, LM Studio's server, or the host's
  kokoro-reader. `up` names it and skips that role.
- **A gated model won't download:** accept its terms on its Hugging Face
  page, log in (`huggingface-cli login`, or `HF_TOKEN`), then
  `python -m services setup` to store the token.
- **SELinux denies a model file:** use `LUNA_MODELS_DIR`, which is mounted
  without relabelling.

## Tests

The unit tests run with the rest: `python -m unittest discover -v tests`.

The container tests use real Podman, and only their own `luna-test-*`
containers, volumes and secrets, on ports 18080-18082 and 18899:

```bash
LUNA_CONTAINER_TESTS=1 python -m unittest discover -v tests/containers
LUNA_CONTAINER_TESTS=full ...       # also the reasoning and vision rows
```

They download a few small public models once, into `luna-test-models`.
`LUNA_TEST_GATED=<user>/<repo>` also tries a gated model with your token.

### From zero

For working on the backend: put this checkout and this machine back as
they were just after `git clone`, then install as a new user would. It
catches what a long-lived setup hides.

```bash
tests/zero/reset.sh                          # show what goes, ask, reset
tests/zero/reset.sh --install                # reset, install (--yes), up, check
tests/zero/reset.sh --install --interactive  # the same, answering the installer yourself
tests/zero/check.sh                          # after any install: does everything answer?
```

It removes, with **no backup**: every gitignored file in the checkout
(`.venv`, `install.json`, `llama/.api_key`, the `.local.env` files,
`llama/models`, deploy's `.env` and keys, `agent/`'s databases), and the
tracked `config.json` and `agent/` go back as committed. It also removes
`~/.cache/ai-voice` (her logs and knowledge index), the `luna` launcher
and app entry, the host voice's venv, and in Podman every `luna-*`
container, volume and network, every `luna_*` secret, our images, and
upstream's llama.cpp and `python:3.12-slim` images. So everything
downloads and builds again.

Source files, committed or not, are never touched. Neither are system
packages, your Hugging Face cache and token, LM Studio's models, the
kokoro-reader clone, or other containers and images. The installer runs
with `--no-system`, so it never needs sudo. `check.sh` changes nothing
but a cache: three test facts' vectors.

## What was checked

On Fedora 43 (SELinux), rootless Podman 5.8.8, an RTX 3070 (8 GB, CUDA 13).

- **Podman 4.7** is the oldest that works: `secret create --replace` is new
  in 4.7, and CDI devices in 4.1. A running container keeps its old copy
  of a replaced secret, even after a restart, so a changed key makes new
  containers. Podman won't make an empty secret, so the token secret is
  only added when there is one.
- **llama.cpp**: `ghcr.io/ggml-org/llama.cpp`, build b11804, pinned by its
  multi-arch digests (`services/images.py`). Every flag used is in
  `llama-server --help`. The cache variable `LLAMA_CACHE` isn't in
  `--help`, but llama-server's README documents it, and its download code
  reads it first, then `HF_HUB_CACHE`; both point at the volume. The image
  has `bash`, `awk` and `curl`, and runs `/app/llama-server`. `-hf` fetches
  a repo's mmproj by itself.
- The default embedding and reranker models resolve, and give a 1024-wide
  embedding and a right ranking. `/health` answers without the key; the
  API doesn't.
- **The voice**: kokoro-reader at a pinned commit; `kokoro` and `misaki`
  0.9.4 need Python 3.10-3.12, so the image is `python:3.12-slim`, pinned.
  torch 2.14.1 from its own index (CPU, or cu126 and cu130 for NVIDIA).
  The CPU image is 1.64 GB, with no CUDA packages. misaki's English model
  is in the image, so the voice works with no network once its model is
  cached. It runs with `--init`, since as PID 1 it wouldn't stop on SIGTERM.
- **Compose**: every file validates with Podman Compose 1.5 and Docker
  Compose v2. An empty `hf_token` file is fine there (compose mounts it),
  and means no token.
