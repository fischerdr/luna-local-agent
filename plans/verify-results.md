# Verify results (moves into docs/containers.md in P8)

Checked on 2026-10-22 on Fedora 43 (x86_64, SELinux permissive), rootless
Podman 5.8.8, an RTX 3070 (8 GB, driver 580.178.04, CUDA 13.0).

## Podman

| Check | Result |
|---|---|
| Minimum version | **4.7.0**. `secret create --replace` first appears in the 4.7 docs; 4.6 doesn't have it. CDI in `--device` needs 4.1. |
| `secret create --replace` while a container holds the secret | Works (exit 0). The running container keeps its old copy, even after `podman restart`, so a changed key must recreate the container. |
| Rootless | `podman info --format '{{.Host.Security.Rootless}}'` gives `true`. |

## llama.cpp image (`ghcr.io/ggml-org/llama.cpp`)

Pinned index digests (multi-arch), all at build **b11804** (commit c7fc509):

| Tag | Digest | Platforms |
|---|---|---|
| `server` | `sha256:8c74db29f1a2bef3f0a331b52c4d6214601dfceed4bee0b94cc48b3b23ff257d` | amd64, arm64, s390x |
| `server-cuda` | `sha256:490d8cd8000203a3ab8fdfce14ef931e2aa85affd4dd4447bcc75f299f00fd9a` | amd64, arm64 |
| `server-cuda13` | `sha256:dfe42bda81c6bd53df864dfcf65e2423e9703f9991c238cfa7aaa893063e2977` | amd64, arm64 |
| `server-rocm` | `sha256:bdbbab3b239f4a7ada461ddb82b01d7328c6988e194572ae4f08144b710f1013` | amd64 |
| `server-vulkan` | `sha256:a644fb651975d9dea49140302577a3b945993fd5efd3400bf89c7d525d03c103` | amd64, arm64 |

The host build (`llama/install.sh`) tracks llama.cpp master (`git clone
--depth 1`, then `pull`), so b11804 is the current build.

| Check | Result |
|---|---|
| Flags in `--help` | `-hf`, `HF_TOKEN`, `--api-key-file`, `--embeddings`, `--reranking`, `--pooling`, `--no-mmproj`, `--mmproj`, `--mmproj-url`, `--cors-origins`, `--slot-save-path`, `--reasoning-format`, `--reasoning-budget`, `--ctx-checkpoints`, `--kv-unified`, `-fa`, `--jinja`, `--alias`, `--metrics`, `--slots`, `-np`, `-ub`: all present. |
| `LLAMA_CACHE` | **Not in `--help`**, but `libllama-common.so` reads it (along with `HF_HUB_CACHE`, `HF_HOME` and `XDG_CACHE_HOME`). Tested: `-e LLAMA_CACHE=/models` with `-hf` puts the download in the volume, in Hugging Face's cache layout (`/models/models--<org>--<repo>/snapshots/...`). |
| Entrypoint | `/app/llama-server`; working directory `/app`; env `LLAMA_ARG_HOST=0.0.0.0`, `LLAMA_ARG_PORT=8080`. Ubuntu 24.04, runs as root. |
| Tools | `bash`, `awk` and `curl` (`/usr/bin/curl`) are present; `wget` isn't. |
| Upstream `HEALTHCHECK` | Dropped when Podman builds in OCI format, which is its default. Health checks go on `podman run --health-cmd` and in compose. |
| `-hf` and mmproj | `--help`: "mmproj is also downloaded automatically if available. to disable, add --no-mmproj". |
| Default `-hf` alias | The full reference (`Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0`). The entrypoint sets `--alias` to the repo name (`Qwen3-Embedding-0.6B-GGUF`). |
| `EMBED_HF` `Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0` | Resolves to `Qwen3-Embedding-0.6B-Q8_0.gguf` (610 MB); embedding size 1024. |
| `RERANK_HF` `ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF:Q8_0` | Resolves and ranks correctly (0.994 for the right fact, against 0.015 and 0.0). |
| `/health` without the key | 200 (it's public). The API endpoints return 401 without the key. |
| Key in `podman inspect` and the logs | Not found in either. |

## Image build

- Built: cpu image 861 MB (`localhost/luna-test-llama:cpu`).
- `LABEL` from an `ARG` came back stale from buildah's cache, so labels are set with `podman build --label`.

## TTS image (`services/tts`)

| Check | Result |
|---|---|
| kokoro-reader | No tags upstream; pinned to commit `7c70135090f4cc228e0fc2d46900cd42d104a392` (2026-10-18). Its server reads `blends.json` beside itself, so blends like `luna` work. |
| kokoro-reader and a key | **It takes none**: no auth, CORS `*`. Locally it's published on 127.0.0.1 only. For a remote server (P7), section 14 says to ask. |
| Kokoro's Python | `kokoro` 0.9.4 `requires_python >=3.10,<3.13`; `misaki` 0.9.4 the same. Narrower than `install.sh`'s 3.10-3.13. |
| Base | `python:3.12-slim` (3.12.15), index digest `sha256:cb33f7f771ecec2ef1225cf39b159591a040391bd73e4c01f00fdfc4c47611d2`. |
| torch | 2.14.1. CPU from `whl/cpu`; `whl/cu126` (CUDA 12 drivers) and `whl/cu130` (CUDA 13) for NVIDIA. PyPI's own Linux wheel pulls CUDA 13. |
| Pins in the image | `pip show`: kokoro 0.9.4, misaki 0.9.4, numpy 2.5.4, torch 2.14.1+cpu; spacy 3.8.16, en_core_web_sm 3.8.0. No nvidia or cuda packages in the CPU image. |
| Size | CPU image **1.64 GB** (the old branch's nvidia/cuda-based image was 11.4 GB). |
| spaCy model | misaki pip-installs `en_core_web_sm` on its first sentence unless it's there; it's now installed at build time, and TTS works with `--network none` once the Kokoro model is cached. |
| SIGTERM | The server ignores it as PID 1 (a 10 s wait, then SIGKILL); `--init` stops it in 0.3 s. |
| Speed (CPU, this machine) | First start ~18 s with the model cached; a short sentence takes ~1.9 s (`af_bella`), ~2.5 s (`luna` blend, first use). |
