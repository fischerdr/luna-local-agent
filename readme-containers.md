# Luna's backend in containers

Her model, the embedding model, the reranker and her voice can run in
rootless Podman containers instead of being built and installed here.
Luna doesn't change: she reaches them at the same addresses with the same
key (`llama/.api_key`). Use it for no llama.cpp build on the host and no
Python for the voice. Or put all four on a GPU server on your network.

The full reference is [docs/containers.md](docs/containers.md).

## What runs where

| Container | Port | Image | Job |
|---|---|---|---|
| `luna-llama` | 8080 | `luna-services-llama` (cpu, cuda, cuda13, rocm, vulkan) | her model, llama.cpp's server |
| `luna-embed` | 8081 | `luna-services-llama:cpu` | embeddings, for recall by meaning |
| `luna-rerank` | 8082 | `luna-services-llama` | the reranker |
| `luna-tts` | 8899 | `luna-services-tts` (cpu, cuda) | her voice, kokoro-reader |

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

or, with Luna installed already:

```bash
python -m services setup
python -m services setup tts
luna
```

## Running it

```bash
python -m services up          # start them (or keep them, if nothing changed)
python -m services down        # stop them; models and secrets stay
python -m services status      # what's running, and answering
python -m services logs chat   # a container's log
python -m services bench       # where the side models go on a small GPU
python -m services reset       # remove what it made (asks first)
```

- **With Luna:** `"services": {"autostart": true}` in config.json, and
  `./start.sh` starts them first and stops them after
  (`"stop_on_exit": false` leaves them running). Setup offers this.
- **Small GPUs:** `EMBED_NGL` and `RERANK_NGL` in
  `services/services.local.env` put the side models on the CPU or GPU;
  `bench` says which.
- **Models:** `MODEL_HF` (and `EMBED_HF`, `RERANK_HF`) from Hugging Face,
  or your own GGUFs with `LUNA_MODELS_DIR` (llama/models, LM Studio's
  folder), or a file already imported into the `luna-models` volume.

## A GPU server on your network

No tunnels, plain HTTP on your LAN. On the server:

```bash
cd deploy && cp .env.example .env       # LUNA_BIND, MODEL_HF
python -m services setup --server --allow <your machine's IP>
podman compose -f compose.yaml -f compose.cuda.yaml up -d
```

On your machine: `./installer/install.sh --remote <the server's IP>`.
The key guards the model servers; the voice has none, so the firewall
rule setup prints is what keeps everyone else out
([details](docs/containers.md#a-gpu-server-on-your-network)).

## Installer flags

- `--backend container`: her model in containers. Example: `./installer/install.sh --backend container`
- `--voice container`: her voice in a container. Example: `./installer/install.sh --voice container`
- `--remote ADDR`: everything on a server on your network. Example: `./installer/install.sh --remote 192.168.1.50`
- `--gpu cpu|cuda|rocm`: which image. Example: `./installer/install.sh --backend container --gpu cpu`

Which flags go together: [docs/install.md](docs/install.md#containers-and-a-remote-server).

## Tests

`python -m unittest discover -v tests` runs the unit tests, with no Podman
needed. `LUNA_CONTAINER_TESTS=1` (or `full`) with
`python -m unittest discover -v tests/containers` runs the container
tests, on their own `luna-test-*` containers.

From zero, for working on it: `tests/zero/reset.sh --install` puts this
checkout and machine back as after `git clone` (models, data, images and
all; no backup), installs as a new user would, and checks it
([details](docs/containers.md#from-zero)).

## Troubleshooting

- Logs: `podman logs luna-llama` (or `luna-embed`, `luna-rerank`, `luna-tts`).
- 401: the keys don't match. Luna reads `llama/.api_key`; run `python -m services up` after changing it.
- No GPU in the container: check the CDI spec (`nvidia-ctk cdi list`).
- Port busy: a host `llama/start.sh` or LM Studio is holding it; `up` names it.
- SELinux denies a model file: use `LUNA_MODELS_DIR`, mounted without relabelling.

More: [docs/containers.md](docs/containers.md), [docs/install.md](docs/install.md).
