# Luna's backend in containers

Her model, the embedding model, the reranker and her voice can run in
rootless Podman containers instead of being built and installed here.
Luna doesn't change: she reaches them at the same addresses with the same
key (`llama/.api_key`). Use it for no llama.cpp build on the host and no
Python for the voice. Or put all four on a GPU server on your network.

It's a folder in `backends/` ([how those plug in](../../docs/development.md#backends)):
the installer offers it on Linux with Podman, and `--remote` anywhere.
Delete the folder and Luna installs and starts exactly as without it.

The full reference is [docs.md](docs.md).

## What runs where

| Container | Port | Image | Job |
|---|---|---|---|
| `luna-llama` | 8080 | `luna-services-llama` (cpu, cuda, cuda13, rocm, vulkan) | her model, llama.cpp's server |
| `luna-embed` | 8081 | `luna-services-llama:cpu` | embeddings, for recall by meaning |
| `luna-rerank` | 8082 | `luna-services-llama` (GPU if there's room, else cpu; off if the CPU is too slow) | the reranker |
| `luna-tts` | 8899 | `luna-services-tts` (cpu, cuda, cuda13) | her voice, kokoro-reader |

All on 127.0.0.1. The introspection lab stays on the host (`llama/install.sh`).

## You need

- Linux, and Podman 4.7 or newer, rootless.
- For NVIDIA: nvidia-container-toolkit and its CDI spec. For AMD: ROCm (`/dev/kfd`) or Vulkan (`/dev/dri`).
- A Hugging Face token only for gated models.

## Quick start

```bash
./installer/install.sh --backend container --voice container
luna
```

With only Python 3.14 (newer than the host voice allows) and Podman, a
plain `./installer/install.sh` offers containers by itself.

or, with Luna installed already:

```bash
python -m backends.containers setup
python -m backends.containers setup tts
luna
```

## Running it

```bash
python -m backends.containers up          # start them (or keep them, if nothing changed)
python -m backends.containers down        # stop them; models and secrets stay
python -m backends.containers status      # what's running, and answering
python -m backends.containers logs chat   # a container's log
python -m backends.containers bench       # where the side models go on a small GPU
python -m backends.containers reset       # remove what it made (asks first)
```

- **With Luna:** `"backends": {"containers": {"autostart": true}}` in config.json, and
  `./start.sh` starts them first and stops them after
  (`"stop_on_exit": false` leaves them running). Setup offers this.
- **Small GPUs:** the GPU goes to chat and her voice first; the reranker
  and embedder get it only if there's room, else the CPU - and the
  reranker is off when the CPU is too slow for it. `bench` measures and
  writes `TTS_DEVICE`, `RERANK_NGL`, `EMBED_NGL` (and `RERANK`) with your OK.
- **Models:** `MODEL_HF` (and `EMBED_HF`, `RERANK_HF`) from Hugging Face,
  or your own GGUFs with `LUNA_MODELS_DIR` (llama/models, LM Studio's
  folder), or a file already imported into the `luna-models` volume.

## A GPU server on your network

No tunnels, plain HTTP on your LAN. On the server:

```bash
cd deploy && cp .env.example .env       # LUNA_BIND, MODEL_HF
python -m backends.containers setup --server --allow <your machine's IP>
podman compose -f compose.yaml -f compose.cuda.yaml up -d
```

On your machine: `./installer/install.sh --remote <the server's IP>`.
The key guards the model servers; the voice has none, so the firewall
rule setup prints is what keeps everyone else out
([details](docs.md#a-gpu-server-on-your-network)).

## Installer flags

- `--backend container`: her model in containers. Example: `./installer/install.sh --backend container`
- `--voice container`: her voice in a container. Example: `./installer/install.sh --voice container`
- `--remote ADDR`: everything on a server on your network. Example: `./installer/install.sh --remote 192.168.1.50`
- `--gpu cpu|cuda|rocm`: which image. Example: `./installer/install.sh --backend container --gpu cpu`
- `--model SIZE`, `--standard`: another model size, or Qwen's own release instead of the uncensored default. Example: `./installer/install.sh --backend container --model 9b-q4`

Which flags go together: [docs.md](docs.md#installer-flags).

## Tests

`python -m unittest discover -v -s backends/containers/tests -t .` runs its unit
tests, with no Podman needed. `LUNA_CONTAINER_TESTS=1` (or `full`) with
`python -m unittest discover -v -s backends/containers/tests/live -t .` runs the container
tests, on their own `luna-test-*` containers.

From zero, for working on it: `backends/containers/tests/zero/reset.sh --install` puts this
checkout and machine back as after `git clone` (models, data, images and
all; no backup - it asks once for the checkout, and group by group for
anything outside it), installs as a new user would, and checks it
([details](docs.md#from-zero)).

## Troubleshooting

- What's up: `python -m backends.containers status`; a role that's off on purpose says so.
- Logs: `podman logs luna-llama` (or `luna-embed`, `luna-rerank`, `luna-tts`).
- 401: the keys don't match. Luna reads `llama/.api_key`; run `python -m backends.containers up` after changing it.
- No GPU in the container: check the CDI spec (`nvidia-ctk cdi list`).
- Port busy: a host `llama/start.sh` or LM Studio is holding it; `up` names it.
- SELinux denies a model file: use `LUNA_MODELS_DIR`, mounted without relabelling.

More: [docs.md](docs.md), and Luna's own [install page](../../docs/install.md).
