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
