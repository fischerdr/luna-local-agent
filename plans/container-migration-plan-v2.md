# Backend services in containers: implementation brief

This brief is for the agent that implements containerised backend services
for Luna, starting from `main`. It's self-contained: follow it in order, meet
every acceptance check, and stop and ask where section 14 says to.

- Base: `main` at `6d1ef9f` or later. That base includes the settings page,
  config/schema.py, the installer's model picker, her built-in manual and
  the first run.
- Replaces nothing on `main`. The older `docs/container-llama-migration-plan.md`
  on `feat/python-314-container` described a different approach. Don't use it,
  and don't use that branch's code.
- This brief lives in `plans/`, not `docs/`. Main's `tools/manual.py`
  (`read_manual`) indexes `README.md` and every `docs/*.md`, so anything in
  `docs/` becomes Luna's own manual. Keep plans and notes about unbuilt
  features out of `docs/`. `docs/containers.md` gets written only once the
  feature exists, and it describes only what was built.

---

## 1. Mission

Add a second, optional way to run Luna's backend services: in rootless
Podman containers on this machine, or with Docker/Podman Compose on a GPU
server on the local network. The backend services are:

- the chat llama-server
- the embedding server
- the reranker
- the Kokoro TTS server

**The main application doesn't change.** Main already reaches every backend
through a URL and a key:

- `llm.llama_url`, `llm.embed_url` and `llm.rerank_url` in `config/server.py`
- `tts.url` (`launch.py:tts_url()`)
- the key in `llama/.api_key` (`LLAMA_KEY_FILE`)

`launch.py` already uses a server it didn't start: it calls `model_up()`
first, and only starts the voice when its mode is `managed` and its URL is
local (`_here()`). So containers that answer on those URLs with that key
need no app changes.

Existing installs, defaults, `llama/start.sh`, LM Studio, the host llama.cpp
build, the host Kokoro venv and the introspection lab must all behave
exactly as they do on `main`.

---

## 2. Hard rules

### 2.1 Branch

```bash
git switch -c feat/containers 6d1ef9f
```

- This repo isn't ours. Never commit to or push `main`, and don't pull: the
  base stays at `6d1ef9f`. All work, this brief included, lives on
  `feat/containers`, which later moves to its own repo.
- Don't merge, rebase onto, or copy code from `feat/python-314-container`.
- The one cherry-pick allowed is `330fd53` (`.flake8`).

### 2.2 Scope fence: what may change

- **Added files**:
  - anything under `services/`, `deploy/` and `tests/`
  - `docs/containers.md`
  - `readme-containers.md` at the repo root
  - `plans/` (this brief) and `.flake8` (the one cherry-pick)
- **Modified files, opt-in edits only**:
  - `installer/install.py`: the new choices and flags in section 7. Defaults
    stay the same.
  - `launch.py`: an autostart block behind `services.autostart`, default
    false (section 6.3).
  - `config.example.json`: one new `services` block, off by default.
  - `.gitignore`: append-only entries for `services/services.local.env`,
    `deploy/.env`, `deploy/api_key` and `deploy/hf_token`.
  - Docs, additive only, exactly as listed in section 11: `README.md`,
    `docs/README.md`, `docs/_Sidebar.md`, `docs/install.md`,
    `docs/configuration.md`, `docs/development.md`, `docs/knowledge.md`,
    `docs/vision.md`, `docs/reasoning.md` and `docs/sysadmin.md`.
- **Every other tracked file stays byte-identical to `main`.** That includes
  `llama/server.env`, `llama/start.sh`, `config/`, `embedmem.py`,
  `lmstudio.py`, `tts/` and every existing test.

Ship the fence as `tests/check_scope_fence.py`. Give it no `test_` prefix,
so unittest discovery skips it; after the merge it would only be comparing
`main` with itself. Run it at every phase gate. It compares against
`BASE=$(git merge-base HEAD main)`, not `main` itself, so a moved `main`
can't leak unrelated changes into the check. It checks four things:

1. `git diff --name-only --diff-filter=MDR $BASE` lists only the modified
   files above.
2. Every added file (`--diff-filter=A`) is under an allowed path.
3. `git diff --numstat $BASE -- docs README.md` shows no deleted lines, apart
   from the single knowledge.md phrase in section 11.
4. The untracked files written at runtime are only `llama/.api_key`, if it
   was missing, and `llama/server.local.env`, with consent. Main already
   gitignores both.

### 2.3 Main keeps working

After every phase:

- `python -m unittest discover -v tests` passes, with **no edits to existing
  tests**. CI runs Python 3.12 offline (`tests/offline.py`), without Podman.
- `./start.sh --check` and `./start.sh` behave as on `main` whenever there's
  no `services` config.

### 2.4 The host stays untouched

- No sudo.
- No host package or pip installs for services.
- No `--network host`.
- Rootless Podman only. Require
  `podman info --format '{{.Host.Security.Rootless}}'` to be `true`.
  Also require a minimum Podman version that supports `secret create
  --replace` and CDI. Find out which version that is, and write it down.
- Rootless Podman is the security boundary. The ggml images run as
  container-root, which maps to the host user. **Don't add `USER`** to the
  llama image: it breaks `/app` and volume writes for no gain.
- Local containers publish only on the `HOST` address from `server.env`,
  which is `127.0.0.1` by default (section 4.3).
- State lives in named volumes (`luna-models`, `luna-data`,
  `luna-tts-cache`) and Podman secrets. Nothing is written into the repo
  except the two untracked files in 2.2.

### 2.5 The installer's promises (from the README and `docs/install.md`)

- Ask before anything that touches the system, and show the exact command
  first. That covers pulls, builds, volumes, secrets and model imports.
- Never run sudo. Print the command instead.
- Re-running repairs: it carries on where it left off and does only what's
  missing.

### 2.6 Verify, don't assume

- Every llama.cpp flag or env var this brief names (`-hf`, `LLAMA_CACHE`,
  `HF_TOKEN`, `--api-key-file`, `--embeddings`, `--reranking`, `--pooling`,
  `--no-mmproj`, `--cors-origins`) must show up in `llama-server --help` in
  the pinned image. If any is missing, stop and ask.
- Everything marked *verify* below must be checked and the result written
  down in `docs/containers.md`.

### 2.7 Commits and style

- One commit per phase, ending with the session's attribution line.
- Match the code around you: the comment density, `logbook` usage and
  naming in `config/server.py`, `launch.py` and `installer/install.py`.
- Run `flake8` with the repo `.flake8`, and `shellcheck` on every shell
  file.

---

## 3. How it plugs into main (no app changes)

| App expects (main) | Container provides |
|---|---|
| chat at `llama_url` :8080, key from `llama/.api_key` | `luna-llama` on `HOST:PORT`, `--api-key-file /run/secrets/api_key`; the secret comes from `llama/.api_key` |
| `embed_url` :8081 and `rerank_url` :8082, same key | `luna-embed` and `luna-rerank`, run when main's `EMBED`/`RERANK` say so |
| `tts.url` :8899 | `luna-tts` on 127.0.0.1:8899 |
| `backend: auto` probes `/props` at startup | the containers are up before Luna starts, so auto picks llama.cpp |

```text
 host (Luna, unchanged)                 rootless Podman
 ─────────────────────                  ───────────────────────────────
 config.json URLs ──── 127.0.0.1:8080 ─▶ luna-llama  (ROLE=chat)
 llama/.api_key   ──── 127.0.0.1:8081 ─▶ luna-embed  (ROLE=embed)
                  ──── 127.0.0.1:8082 ─▶ luna-rerank (ROLE=rerank)
 tts.url          ──── 127.0.0.1:8899 ─▶ luna-tts
                                         volumes: luna-models,
                                         luna-data, luna-tts-cache
                                         secrets: luna_api_key,
                                         luna_hf_token
```

- The **introspection lab** (`introspection/lab.py`, `blind.py`,
  `make_vectors.py`, `read.py`) needs the host build (`llama/install.sh`). It
  isn't containerised. Leave it unchanged and say so in the docs.
- A **remote model** keeps working as main documents it: copy the server's key
  into `llama/.api_key`.

---

## 4. Target

### 4.1 The `services/` package and CLI

`python -m services setup|up|down|status|logs <role>|bench|reset`

It uses only the standard library and `requests`, which is already in
`installer/requirements.txt`, so it adds no new runtime dependencies. Import
it lazily and behind guards from `launch.py` and `install.py`, so Windows,
macOS and hosts without Podman never load it.

| File | Job |
|---|---|
| `services/__main__.py` | the CLI |
| `services/settings.py` | resolves settings (4.2) and models (4.4) |
| `services/images.py` | `image_for_host()` (4.7), pulls, builds, image identity (4.6) |
| `services/run.py` | `run_args(role, settings)`, a pure function that returns the argv |
| `services/lifecycle.py` | `ensure`, `stop`, `status`, `wait_ready`, port checks |
| `services/bench.py` | the small-GPU recommendation (section 8) |
| `services/services.env` | the new settings, with defaults |
| `services/llama/Dockerfile`, `entrypoint.sh` | the llama image (section 5) |
| `services/tts/Dockerfile` | the TTS image (section 5) |

`wait_ready(url, timeout)` polls `/health`. Never wait with a fixed `sleep`.

### 4.2 Settings

Let bash resolve them, the way `start.sh` does. **Don't write a Python
`.env` parser.** Run this from `llama/`:

```bash
env -i bash -c 'set -a
  source "${LLAMA_ENV:-server.env}"
  [ -f server.local.env ] && source server.local.env
  source ../services/services.env
  [ -f ../services/services.local.env ] && source ../services/services.local.env
  env -0'
```

- Keep only the keys you know.
- `LLAMA_ENV` *replaces* `server.env`, as it does in main's `start.sh`.
- `server.mac.env` isn't involved, because containers are Linux-only.
- Write an unquoted `--env-file` to `$XDG_RUNTIME_DIR`, or a temp dir. Never
  write it into the repo. It holds no secrets.

**Where keys live:**

- **Main's switches stay the switches.** `EMBED` and `RERANK`
  (`auto|on|off`) decide whether `luna-embed` and `luna-rerank` run.
  `MMPROJ=off` turns vision off. `CTX`, `NGL`, `PARALLEL`, `KV_UNIFIED`,
  `REASONING_*`, `ALIAS`, `EXTRA`, the ports and `HOST` all work as in
  `start.sh`. `auto` means "run it if a model resolves".
- **New keys go only in `services/services.env`**, overridable by
  `services/services.local.env`: `MODEL_HF`, `MMPROJ_HF`, `EMBED_HF`,
  `RERANK_HF`, `EMBED_NGL`, `RERANK_NGL`, `LUNA_MODELS_DIR`,
  `EMBED_ONDEMAND` and the idle times. Never add a second on/off switch.
- Defaults:
  - `EMBED_HF="Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0"`
  - `RERANK_HF="ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF:Q8_0"`

  These are the models the installer already downloads. *Verify* both
  quant tags.
- `MODEL_HF` has no default. Setup asks for it, or uses a local file.

### 4.3 Publish address and ports

- Inside the container, llama-server listens on `0.0.0.0` on a fixed port.
- The publish address is `HOST` from `server.env` (main defaults it to
  `127.0.0.1`). If `HOST` isn't loopback, `services up` warns and asks once.
  That matches main's documented "set HOST=0.0.0.0 to share".
- Host ports come from `PORT`, `EMBED_PORT` and `RERANK_PORT`. TTS uses 8899.
- **Port check before `up`.** If something other than our container holds a
  port (a host `llama/start.sh`, LM Studio, another app), say what it is
  (use `ss -ltnp` when available) and skip that role. Don't surface Podman's
  raw error.

### 4.4 Model resolution (per role, settled at `setup`/`up`, never first found at runtime)

1. If `*_HF` is set, use `-hf` into the `luna-models` volume with
   `LLAMA_CACHE=/models`.
2. If `MODEL` (or `EMBED_MODEL`/`RERANK_MODEL`) names a file:
   - A host path under `LUNA_MODELS_DIR` becomes `/models-host/<relative
     path>`, keeping any subfolders.
   - A path starting with `/models/` means an imported file in the volume.
   - Any other host path is an error that names the folder to mount.
3. If a `MODEL_MATCH`/`*_MATCH` pattern is set, search `/models-host`, then
   `/models`, using `start.sh`'s rules: `-iname`, skip mmproj files, and
   treat more than one match as an error.
4. If nothing resolves:
   - chat: `up` refuses and says how to fix it;
   - embed and rerank: `auto` skips the role with one line, as `start.sh`
     does, and `on` refuses.

`LUNA_MODELS_DIR` is optional. It's mounted read-only at `/models-host`, so
LM Studio's GGUFs, or an existing `llama/models`, get used without a second
copy. On SELinux it needs `--security-opt label=disable` on that container.
**Never relabel the host folder.**

**Vision:**

| Setting | Flag |
|---|---|
| `MMPROJ=off` | `--no-mmproj` |
| `MMPROJ_HF` set, or the mmproj file resolves | `--mmproj …` |
| neither | `-hf`'s own mmproj, or the same folder search `start.sh` does |

*Verify* the flag names and how `-hf` handles an mmproj.

**Alias:** the GGUF file name, as on main. For `-hf` models it's the repo
name. Either way the model name in the UI stays stable.

### 4.5 Secrets

- **One path inside every container: `/run/secrets/api_key` and
  `/run/secrets/hf_token`.**
  - Local: `--secret luna_api_key,target=api_key`. Add
    `--secret luna_hf_token,target=hf_token` only when a token exists, since
    naming a missing secret fails the run.
  - Compose: `secrets: [{source: api_key, target: api_key}]`, and the same
    pattern for `hf_token`.
  - Secret names use underscores everywhere.
- **The API key** comes from `llama/.api_key`. If that file is missing,
  create it with main's logic (`umask 077`; urandom, then base64).
- **The HF token** belongs to the user. Look for it in Hugging Face's own
  order: `HF_TOKEN`, then `$HF_HOME/token`, then
  `~/.cache/huggingface/token`. Only read it, never copy it.
  - An env token is passed to `secret create` on stdin.
  - Don't bind-mount the token file; on SELinux that would need a relabel.
- The entrypoints export `HF_TOKEN` only when `/run/secrets/hf_token` is
  non-empty.
- Never put the key or the token in argv, an image, an env-file or a log.
- Refresh a secret with `secret create --replace` only when its
  fingerprint changes (4.6), then recreate the affected containers.
  *Verify* that `--replace` works while an old container still holds the
  secret. If it doesn't, stop that container first.
- The token is optional. Public models download without one. A gated
  download that fails says where to put the token (`huggingface-cli login`).

### 4.6 Container and image identity

- Run containers with `--replace` and a label
  `luna.config-hash=<sha256>`. The hash covers:
  - the argv;
  - the env-file contents;
  - the image digest;
  - the SHA-256 fingerprints of the API key and the HF token (never the
    values).

  A changed setting, key, token or image therefore recreates the
  container. That's required, because Podman copies a secret into a
  container when it's created.
- **Identify images by label, not by tag.** Every image we build carries
  `luna.image-version=<services version>` and `luna.base-digest=…`, and
  `ensure` only uses an image whose labels match. The old branch left a
  bare retag, `localhost/luna-llama:cuda13`, with no entrypoint, plus
  `luna-tts:container_cuda`. A check by tag alone would silently reuse
  them.
- **Migration:** a `luna-*` container with no `luna.config-hash` label is
  from the old branch. `setup` lists any it finds and asks before replacing
  them. It never removes them silently.

### 4.7 Image choice and GPU passthrough

`image_for_host()` picks the backend in this order:

1. `nvidia-smi` works and a CDI spec exists: `server-cuda13` when the
   driver's "CUDA Version" major is 13 or more, otherwise `server-cuda`.
2. `nvidia-smi` works but there's no CDI spec: **cpu**, and print the CDI
   command. Never fall to Vulkan, because Vulkan on NVIDIA needs the ICD
   inside the container.
3. `/dev/kfd` exists: `server-rocm` on amd64. On arm64, use `server-vulkan`.
4. `/dev/dri` exists: `server-vulkan`.
5. Otherwise: `server` (cpu).

`--gpu`, or the `LLAMA_BACKEND` setting, overrides the result.

**Base images: `ghcr.io/ggml-org/llama.cpp`** (research of 2026-10-20; tags
move, so confirm each one in the registry):

| Tag | Platforms | Notes |
|---|---|---|
| `server` | amd64, arm64 | cpu |
| `server-cuda` | amd64, arm64 | CUDA 12 |
| `server-cuda13` | amd64, arm64 | the image sets `NVIDIA_REQUIRE_CUDA=cuda>=13.4` |
| `server-rocm` | amd64 | |
| `server-vulkan` | amd64, arm64 | AMD and Intel through Mesa |
| `server-intel` | | SYCL; not used by default |

- Pin each tag by digest, and record its llama.cpp build number.
- Upstream says the GPU images are build-tested only, never run-tested.
  That's why the GPU container tests and digest pins are required.
- For a backend that has no working image, build locally from upstream's
  `.devops/*.Dockerfile`, or stop and ask.

**Run flags.** These were proven on Fedora with SELinux, on rootless Podman
5.8:

| Backend | Flags |
|---|---|
| NVIDIA | `--device nvidia.com/gpu=all --security-opt label=disable` |
| ROCm | `--device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --security-opt label=disable` |
| Vulkan | `--device /dev/dri --group-add video --security-opt label=disable` |

- ROCm and Vulkan also get `--group-add render` when that group exists on
  the host.
- ROCm: warn if `rocminfo` shows a gfx target the image doesn't include.
- `label=disable` is only needed on SELinux hosts.

**CDI specs:**

- The spec lives in `/etc/cdi/nvidia.yaml` or
  `$HOME/.config/cdi/nvidia.yaml`. The second needs
  `--cdi-spec-dir $HOME/.config/cdi`.
- When neither exists, print the user-level command
  `nvidia-ctk cdi generate --output=$HOME/.config/cdi/nvidia.yaml` and
  *verify* it works without sudo. If it doesn't, print
  `sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml` instead. Then
  offer cpu.
- Never use `--gpus all` or `--runtime nvidia` with Podman. They're Docker
  syntax. The one exception is the Docker compose override in 4.8.

**Alternatives** from the research: llama-swap (one endpoint, models
hot-swapped) and RamaLama (Podman GPU auto-detection). Neither is a default.
RamaLama is a host dependency, and llama-swap breaks the app's three-port
contract. Bring them up only through section 14.

### 4.8 `deploy/`: a remote GPU server on the local network

No SSH or tunnels anywhere. Access is plain HTTP on the LAN, protected by
the API key plus a firewall rule that admits a single client IP.

- `compose.yaml` holds the base (cpu) setup:
  - the services `llama`, `embed`, `rerank` and `tts`, with the same images
    and roles as the local containers, always on, `restart: unless-stopped`;
  - named volumes;
  - secrets as files: `./api_key`, and `${HF_TOKEN_FILE:-./hf_token}`;
  - ports `"${LUNA_BIND:?set LUNA_BIND to this server's LAN IP}:<port>:<port>"`,
    so compose refuses to start without an explicit address.
- Compose fails when a secret file is missing. `services setup --server`
  therefore creates an **empty** `./hf_token` placeholder when there isn't
  one, and the entrypoint ignores an empty token.
- Health:
  - If the llama image has `curl` (*verify*), add a `healthcheck`.
  - If not, leave the healthcheck out. The client's probe is the real
    check either way.
- GPU overrides:
  - **Podman**: `compose.cuda.yaml` uses CDI
    `devices: [nvidia.com/gpu=all]` with `security_opt: [label=disable]`.
    `compose.rocm.yaml` and `compose.vulkan.yaml` use the flags from 4.7.
  - **Docker**: `compose.cuda.docker.yaml` uses `deploy.resources.reservations.devices`
    (`driver: nvidia`, `capabilities: [gpu]`). The server needs
    nvidia-container-toolkit for this.
- `.env.example`: `LUNA_BIND`, the `*_HF` models, the image tags,
  `HF_TOKEN_FILE`, and a one-liner that generates `api_key`.
- `services setup --server --allow <client-ip>` **prints** (never runs) a
  firewall rule that admits only that client:
  - firewalld:
    `sudo firewall-cmd --permanent --add-rich-rule='rule family=ipv4 source address=<client-ip> port port=8080-8082 protocol=tcp accept'`,
    the same rule for 8899, then `--reload`;
  - ufw: `sudo ufw allow from <client-ip> to any port 8080:8082 proto tcp`,
    plus 8899;
  - nftables: the equivalent rule.
- The docs say plainly that this is plain HTTP on a trusted LAN. The key
  travels unencrypted, and the IP allow-list is the second layer.
- **Drift guard:** `tests/test_compose_drift.py` checks `compose.yaml`
  against `run_args()`: ports, env keys, image, role and secret paths.

---

## 5. Image rules (lessons from the branch, where an agent had to tweak things to get them working)

### 5.1 llama image: `services/llama/Dockerfile`

```dockerfile
ARG BACKEND=server
FROM ghcr.io/ggml-org/llama.cpp:${BACKEND}@sha256:<pinned per backend>
```

- The upstream entrypoint is `/app/llama-server`. Have `entrypoint.sh` call
  it by that absolute path. The branch's `exec llama-server` depended on
  `PATH`, and its `Dockerfile.cuda` never actually ran.
- The image has `bash` and `awk`; the branch used both. *Verify* whether it
  has `curl`. If it doesn't, have `services` poll from the host and skip
  `HEALTHCHECK`.
- `entrypoint.sh` takes `ROLE=chat|embed|rerank` and builds the same
  arguments main's `start.sh` does:
  - chat: alias, mmproj, `-c`, `-ngl`, `-t`, `-b`, `-ub`, `-np`,
    `--ctx-checkpoints`, `--kv-unified`, `-fa auto`, `--jinja`, the
    reasoning flags, `--metrics --slots --slot-save-path /data/slots`, and
    `EXTRA`;
  - embed: `--embeddings --pooling $EMBED_POOLING`, ctx, batch and ubatch,
    as in `start.sh`, plus `-ngl $EMBED_NGL`;
  - rerank: `--reranking --pooling rank`, ctx, batch and ubatch, plus
    `-ngl $RERANK_NGL`;
  - every role: `--host 0.0.0.0 --api-key-file /run/secrets/api_key
    --cors-origins localhost`.
- It runs as container-root. Don't add `USER`.

### 5.2 TTS image: `services/tts/Dockerfile`

Each rule below fixes a real bug in the branch's `tts/Dockerfile*`:

1. **Quote every version pin** (`"kokoro>=0.9.4" "numpy>=1.24"`), or use a
   pinned requirements file. In a shell-form `RUN`, the branch's unquoted
   `kokoro>=0.9.4` was read as a shell redirect: the pins were lost and junk
   files were written.
2. **Install CPU torch from its own index, as a separate step**:
   `pip install --index-url https://download.pytorch.org/whl/cpu
   "torch==<pin>"`. `"torch[cpu]"` isn't a real extra; it pulled the CUDA
   wheel from PyPI. For CUDA, use `ARG TORCH_INDEX=…/whl/cu12x`.
3. **Use one base, `python:3.12-slim`, for both CPU and CUDA.** The branch's
   `nvidia/cuda:…cudnn-runtime` base plus full torch came to 11.4 GB,
   because torch wheels already bundle the CUDA runtime, and CDI injects the
   driver anyway. Record the new image size.
4. **No EXTERNALLY-MANAGED workarounds.** The slim base's pip isn't marked
   managed, so drop `--break-system-packages` and the `rm`.
5. **Pin kokoro-reader:** `ARG KOKORO_REF=<tag or sha>`, then clone and
   check out that ref. The branch's `--depth 1` clone was unpinned, and its
   copy loop for `blends.json`/`start_server.sh` was dead code; don't carry
   it over.
6. **`HF_HOME=/cache` on the `luna-tts-cache` volume**, so the roughly
   400 MB model downloads once, not once per container.
7. `KOKORO_HOST=0.0.0.0` is fine **inside** the container, because the
   publish address is `127.0.0.1:8899`. The branch's `--network host`
   exposed it on the LAN. Take `KOKORO_VOICE` from `config.json`'s `voice`,
   not a hard-coded `am_adam`.
8. Use CUDA for TTS only when CDI is present, with the NVIDIA flags from
   4.7. ROCm torch for TTS is out of scope, so AMD hosts get CPU TTS.

### 5.3 Both images

- The build context is `services/<image>/` only, with a `.containerignore`,
  and never the repo root. That keeps `config.json`, keys, `agent/` and the
  models out of every image.
- `tests/containers/test_image_build.py` checks:
  - the context exclusions;
  - that every pin survived (`pip show kokoro` reports the pinned version);
  - that the CPU TTS image has no `nvidia` or `cuda` packages;
  - that the image sizes are recorded.

---

## 6. Logic flows

### 6.1 Install (`./installer/install.sh`)

```text
1. Checks              Python, GPU, disk                        (unchanged)
2. System packages     shows the exact command, asks            (unchanged)
3. Luna's venv         .venv or ~/ai-voice-venv                 (unchanged)
4. Her voice           install | url:… | skip | container       (+ container)
                         container → services setup tts
5. Her brain           lmstudio | llama | container | remote | skip
                         container → services setup
                         remote    → section 7.1
6. Extras                                                       (unchanged)
7. Launchers                                                    (unchanged)
8. Health check        + container lines: image, running, healthy, key
```

### 6.2 `services setup` (asks before each step and shows the command)

```text
 1. Podman present, rootless, recent enough?
      no → say how to get it; offer LM Studio or the host build; stop
 2. GPU backend (4.7); no CDI → print the command, offer cpu
 3. llama/.api_key missing → create it (main's logic);
      secret luna_api_key
 4. HF token found? (4.5) → secret luna_hf_token, or "public models only"
 5. Old luna-* containers without labels? (4.6) → list them, ask
 6. Pull or build the images (pinned digests, labels)
 7. Create the volumes
 8. Models (4.4):
      llama/models has GGUFs → offer LUNA_MODELS_DIR=llama/models (read-only)
      or an LM Studio folder, an import, or -hf on first up (size shown)
 9. Voice container → install.json voice = external, url 127.0.0.1:8899
      (so launch.py never also starts the host Kokoro venv)
10. bench (section 8) → recommend settings, write only with consent
11. Autostart? → config.json "services": {"autostart": true}
```

### 6.3 Run (`luna` / `./start.sh` → `launch.py`)

```text
1. Package check                                                (unchanged)
2. services.autostart set? → services up:
     refresh secrets if their fingerprints changed
     per role: same config hash → keep, else run --replace
     wait_ready with progress (long on the first -hf download;
       the timeout is configurable; Ctrl+C skips the wait)
     NEVER blocks the start: if a role is down, print one line and go on
     remember what it started
3. start_llama()   skipped if step 2 started or kept a chat container,
                   even one still loading; say "still loading in its
                   container". Otherwise unchanged.
4. Voice           skipped if step 2 handles tts. Otherwise unchanged.
5. Luna starts: auto finds llama.cpp via /props + llama/.api_key (unchanged)
6. Turns use :8080/:8081/:8082/:8899; main's fallbacks apply (unchanged)
7. Exit: main's stops, plus services down for what step 2 started,
   if services.stop_on_exit (default true). Volumes and secrets stay.
```

`services up tts` refuses while `install.json` says the voice is `managed`
(a host venv on :8899). It tells the user to run `services setup tts`.

### 6.4 Remote

```text
Server:  clone, deploy/.env (LUNA_BIND, models), deploy/api_key,
         services setup --server --allow <client-ip>   (prints firewall rule)
         podman|docker compose -f compose.yaml -f compose.<gpu>.yaml up -d
Client:  ./installer/install.sh --remote <server LAN IP>
         luna   → model_up() is true and the voice isn't local, so nothing
                  is started here
```

---

## 7. Installer

### 7.1 Remote services from a single address

- There's a new brain choice, `remote`, and a new flag,
  `--remote <ip-or-host>`.
- Validate the address with `ipaddress` or a hostname check. IPv6 goes in
  brackets inside URLs. Reject loopback: a local server is the `container`
  choice.
- Write the config with main's `set_config()`. All of these keys already
  exist on main:
  - `llm.backend = "llama"`
  - `llm.llama_url = http://<host>:8080/v1/chat/completions`
  - `llm.embed_url = http://<host>:8081/v1/embeddings`
  - `llm.rerank_url = http://<host>:8082/v1/rerank`
  - `tts.url = http://<host>:8899`
  - `llm.autostart = false`. Main's `start_llama()` doesn't check whether the
    URL is local, so without this a host with a llama.cpp build would start a
    useless local server whenever the remote one is down.
- Key: ask for the server's `api_key`, pasted text or a path. Write it to
  `llama/.api_key` with `umask 077`, and ask before replacing an existing
  key.
- Voice: set the state to `external`.
- Probe `/health` on all four services with the key. Report the result in
  main's ✓/✗ style:
  - 401: "key doesn't match the server's".
  - No answer: "check that the server is up, that its LUNA_BIND is this
    IP, and that its firewall allows this machine (<client IP>)".

### 7.2 Option contract

Start from **main's** `install.py`. Every existing flag keeps its meaning
on main, and new values are additions only. Note that `--backend llama`
still builds llama.cpp on the host.

| Flag | Values | Notes |
|---|---|---|
| `--backend` | `lmstudio` \| `llama` \| `container` \| `remote` \| `skip` | no flag: main's menu, default `lmstudio` |
| `--remote ADDR` | IPv4, IPv6, hostname | on its own, implies `--backend remote` |
| `--voice` | `install` \| `container` \| `url:http(s)://…` \| `skip` | free-form, as on main (no argparse `choices`, so `url:` works); checked in code. **Default `None`** means "not given", and maps to main's `install` path |
| `--gpu` | `auto` \| `cpu` \| `cuda` \| `rocm` \| `mps` | host voice: as on main. Containers: picks the image. `mps` with any container is an error |
| `--extras`, `--yes`, `--no-system`, `--check` | as on main | |
| `--llama` | **don't add it** | the branch's version silently overrode `--backend` |

All of this lives in one pure function, `validate_args(args) -> (args,
errors, notes)`, called right after `parse_args()`. Errors go through
`ap.error()`. Notes are printed once.

1. `--remote` with a `--backend` other than `remote` is an error.
2. `--backend remote` without `--remote` under `--yes` is an error.
   Interactively, ask for the address.
3. `--backend container` or `--voice container` on a host that isn't Linux
   is an error that suggests `remote` or LM Studio. A missing Podman isn't
   an error: setup explains it and offers main's choices.
4. A `--voice` value outside the table is an error. Main used to fall
   through to `install` on a typo. This is the only place behaviour gets
   stricter.
5. `--backend container|remote` with `--extras embeddings|reranker` drops
   those two extras, with a note: containers use `-hf`, and the remote
   server holds its own models. `lmstudio` and `llama` keep main's handling.
6. Unknown `--extras` names get a note that lists them. Main used to drop
   them silently.
7. `--gpu cuda` with no NVIDIA GPU, or `--gpu rocm` with no `/dev/kfd`,
   gets a note, then the cpu image.
8. `--backend remote` without `--voice` points the voice at the remote
   :8899. An explicit `--voice` always wins.
9. `--check` with other flags runs only the health check, plus one note
   listing the flags it ignored.
10. `--yes` never changes one of main's defaults. The only exceptions are
    the two cases in 7.3.
11. Repeated flags: argparse's last value wins, as on main. Document this,
    and add no special handling.

Also:

- Always check for Podman with `shutil.which("podman")` first.
- Every `input()` handles `EOFError` the way main's `ask()` and `choose()`
  do.
- `state["backend"]` records exactly what was chosen.
- `health()` reports every voice mode.

### 7.3 Voice: when the installer suggests the container

Change `voice()` only, and only where main's default would otherwise fail:

1. An explicit `--voice` is done exactly as given (unchanged).
2. Something already answering on :8899 gets used (unchanged).
3. If Podman is present **and** the installer's Python is outside
   Kokoro's supported range, the default becomes `container`. It still
   asks. Read that range from the pinned kokoro-reader requirements and
   the `kokoro`/`misaki` `requires-python` (*verify*); don't hard-code a
   guess. This only happens when `install.py` is run directly, since
   `install.sh` already limits Python to 3.10–3.13.
4. Otherwise the default stays the host venv, as on main. If that pip
   install fails and Podman is present, offer the container instead of
   ending with "voice server packages failed".
5. With no Podman, main's behaviour stays as it is, plus a one-line hint.

The container fixes Kokoro's Python, not Luna's. Luna's own venv still
needs 3.10–3.13. Widening that range is out of scope (section 14).

---

## 8. Small GPUs (for example 8 GB)

- Each side model is 0.6B and needs under 1 GB of VRAM (per main's
  `server.env`).
  - With a 4B Q8 chat model, chat, embed and rerank fit in 8 GB together.
  - With a bigger chat model or a large `CTX`, they don't.
- Main runs the side servers with the chat `NGL=99`. Here each role is
  placed separately:
  - **The embedder defaults to CPU** (`EMBED_NGL=0`) and always runs.
    Queries are a few dozen tokens, and fact embeddings are cached in the
    database, so it costs no VRAM and stays well inside `embedmem.TIMEOUT`
    (2.0 s).
  - **The reranker goes on the GPU** when chat plus reranker fit with about
    10% to spare. Otherwise it goes on the CPU, if a 24-candidate rerank
    (main's `RERANK_CANDIDATES`) takes under about 1.5 s. Otherwise it's
    off, as on a Mac. A slow reranker can never break a turn: on a timeout,
    main keeps the embedding order.
- `python -m services bench` reads VRAM with `nvidia-smi
  --query-gpu=memory.total,memory.used` (or `rocm-smi`), then times one
  embedding and a 24-candidate rerank on each device. It prints its
  recommendation and writes it only with consent:
  - `EMBED_NGL` and `RERANK_NGL` go to `services/services.local.env`;
  - `RERANK=off` goes to main's own `llama/server.local.env`.

  `setup` runs it once.
- Acceptance, on an 8 GB card with the 4B chat model:
  - after `up`, VRAM is within budget;
  - recall works;
  - there are no embed timeouts in the `embedmem` log.
- True on-demand start and stop is P9. It's off by default and needs the
  user's approval. Don't build a podman-socket proxy to avoid the app
  hook; stop and ask.

---

## 9. Phases

Each phase is one commit. Its gate:

- the unit suite is green;
- that phase's container tests are green locally;
- `tests/check_scope_fence.py` passes.

| Phase | Work | Accept |
|---|---|---|
| P1 | llama image and `entrypoint.sh` (sections 5.1 and 4.4) | cpu image builds; `/health` answers; 401 without the key; shellcheck clean |
| P2 | `services` CLI: `setup`, `up`, `down`, `status`, `logs` for chat, embed and rerank; secrets; identity; port checks | unit tests (section 12); `python -m services up`, then **unmodified** `./start.sh` shows "· llama.cpp" with recall working; `ss -ltn` shows only 127.0.0.1; editing `server.local.env` recreates the container |
| P3 | TTS image and the `tts` role | unmodified Luna speaks through `tts.url`; no host venv is created |
| P4 | HF token (4.5) | a gated model downloads with the token and fails cleanly without it; the token is never in `podman inspect` or the logs |
| P5 | installer (section 7): `validate_args`, `container`, `remote`, `--remote`, `--voice container` | `install.py --check` output on a main-style install is unchanged; the option matrix passes |
| P6 | `launch.py` autostart (6.3) | behaviour unchanged when the key is absent; with it set, `./start.sh` brings everything up; no double start |
| P7 | `deploy/` compose (4.8) | `docker compose config` and `podman compose config` both validate; 4 healthy services; the client with `--remote` works and starts nothing locally |
| P8 | `services reset` and the docs (section 11) | `reset` removes only `luna-*` containers, images, volumes and secrets, and asks before removing `luna-models`; link and command checks pass |
| P9 | *optional, ask first*: on-demand embed and rerank | a default-off hook in `embedmem.available()`/`reranker_available()` calls `services.ensure_async(role)` without blocking (the turn falls back as on main); an idle reaper stops it later; only for local URLs with `EMBED_ONDEMAND` on; tests show the default path is unchanged |

---

## 10. Feature parity

Docs with no backend dependency don't change: configuration, controls,
devices, email, home, moods, plugins, textures, chat, files, reminders,
games and portrait.

These must behave exactly as they do with the host `start.sh`:

| Doc / feature | App code on main | Container must | Parity test (`tests/containers/test_parity.py`) |
|---|---|---|---|
| reasoning: token confidence | `llm/prompts.py` and `llm/streaming.py` send `n_probs` and read `logprobs` | the same llama.cpp behaviour | streamed chat returns per-token probabilities |
| reasoning: adaptive thinking, second look, chores | `llm/turns.py`, `llm/second_look.py`, `lmstudio.py` and `reflect.py` set `reasoning_budget` per request | `--reasoning-format`/`--reasoning-budget` from `server.env`; a llama.cpp build at least as new as `llama/install.sh` builds (record the build number) | budget 0 gives empty `reasoning_content`; -1 gives some |
| vision | `lmstudio.py` reads `/props` → `modalities.vision` | the mmproj rules in 4.4 | vision true with an mmproj, false with `MMPROJ=off` |
| tools | model template | `--jinja` | a tool call comes back in `tool_calls` |
| model name | `lmstudio.label()` | the `ALIAS` rules | the model id equals the expected alias |
| key and CORS | `config/server.py` uses `LLAMA_KEY_FILE` | `--api-key-file`, `--cors-origins localhost` | 401 without the key; CORS is limited to localhost |
| slots | `--slot-save-path` | `/data/slots` on `luna-data` | a save and restore round-trips |
| "Short on memory?" table | `EMBED`/`RERANK`/`MMPROJ`/`CTX`/`PARALLEL` | every switch honoured | `off` means the role doesn't run; `/props` shows CTX and PARALLEL |
| memory, knowledge, web, gestures, portrait, games, learning, tool routing | `embedmem.py` on :8081/:8082, with fallback | `--embeddings --pooling last`, `--reranking --pooling rank` | the embed dimension is right; rerank orders correctly; unmodified `embedmem` works |
| TTS contract | `POST /tts`, `GET /health`, `GET /voices` | kokoro-reader as it is | all three respond; the WAV parses; the configured voice is accepted |
| introspection lab | host build | **not containerised** | none; the docs say so |

`tts/` (main's chatterbox and qwen3 servers) stays untouched. The Kokoro
container lives in `services/tts/`.

---

## 11. Documentation

Rule: every doc that names a backend, a port, a backend file or backend
code gets accurate text for both ways of running. Host wording stays as on
main, and the container wording is added next to it. Keep the house tone:
plain words, short sentences.

### 11.1 New: `readme-containers.md` (repo root, outside `docs/`)

A short overview, one or two screens:

1. What it is: backend services in containers, with Luna unchanged (same
   URLs, same key). Why you'd use it: no host build, no host venv, and a
   remote GPU server option.
2. What runs where: the four containers, each with its name, port, image
   and job. The introspection lab stays on the host.
3. Requirements: rootless Podman (minimum version), NVIDIA CDI or ROCm
   devices for GPUs, and the optional HF token.
4. Quick start: `./installer/install.sh --backend container --voice
   container`, or `python -m services setup`, then `luna`.
5. Running options:
   - `python -m services up|down|status|logs <role>|bench|reset`;
   - autostart (`services.autostart`, `services.stop_on_exit`);
   - small GPUs (`EMBED_NGL`, `RERANK_NGL`);
   - models (`*_HF`, `LUNA_MODELS_DIR`, import).
6. Remote GPU server on the LAN, with no tunnels (section 6.4).
7. The container-related installer flags, each with an example, and a link
   to the conflict rules.
8. Tests: the unit suite, and `LUNA_CONTAINER_TESTS=1|full`.
9. Troubleshooting:
   - logs: `podman logs luna-*`;
   - 401: the keys don't match;
   - no GPU: check CDI;
   - port busy: a host `start.sh` or LM Studio is holding it;
   - SELinux: use `LUNA_MODELS_DIR`.
10. Links: `docs/containers.md` for the full reference, and
    `docs/install.md`.

### 11.2 New: `docs/containers.md`

The full reference:

- every setting;
- the parity table;
- security notes;
- the *verify* results;
- how to run the container tests.

`readme-containers.md` links to it rather than repeating it.

### 11.3 Edits to existing docs (additive)

| File | Change |
|---|---|
| `README.md` | under *Powered by*, add "(built here, or in containers: readme-containers.md)"; one Quick start line; a Documentation link to `docs/containers.md` |
| `docs/README.md`, `docs/_Sidebar.md` | a "Containers" entry after Install |
| `docs/development.md` | add to the tree: `services/` (each file), `deploy/`, `readme-containers.md`, `tests/containers/`; a `llama/` note ("host build; the containers read the same server.env"). Add the new unit tests and opt-in container tests to *Tests*, and `python -m services status` to *Extras* |
| `docs/install.md` | the `container`/`remote` choices in steps 4 and 5; the new flags; an "Or in containers" subsection under *Or llama.cpp directly*; the compose server and `--remote` under *The model on another machine*; the voice container under *Running the Kokoro Server*; `EMBED_NGL`/`RERANK_NGL` under *Short on memory?* |
| `docs/configuration.md` | a `"services"` block line with a link to `docs/containers.md` |
| `docs/knowledge.md` | "(`llama/start.sh` starts it)" becomes "(`llama/start.sh` starts it, or the `luna-embed` container)". This is the one allowed line change |
| `docs/vision.md` | after *Or llama.cpp directly*: the same `MMPROJ` rules apply in containers |
| `docs/reasoning.md` | the lab needs the host build; with containers, run `python -m services down chat` before starting it |
| `docs/sysadmin.md` | one line: with containers, the logs are `podman logs luna-llama` and `podman logs luna-tts` |
| memory, web, tools, gestures, portrait, games, learning | no change. Their generic "embedding server" or `:8081` wording stays true. Re-check each with grep, and edit one only if its wording becomes false |

Checks:

- `tests/test_doc_links.py`: every relative link in `README.md`,
  `readme-containers.md` and `docs/*.md` resolves.
- Every command in `readme-containers.md` exists, either as a
  `python -m services` subcommand or as an `install.py` flag.

---

## 12. Tests

### 12.1 Unit tests (offline, no Podman, run in the existing CI)

Each file is `tests/test_*.py`, imports `tests.offline` first (as
`tests/test_config.py` does), and mocks subprocess and HTTP. Files go to
temp dirs with a temp `HOME`/`HF_HOME`.

| File | Covers |
|---|---|
| `test_services_settings.py` | layering (`LLAMA_ENV` or `server.env`, then `server.local.env`, `services.env`, `services.local.env`); bash quoting and comments; all four resolution steps (subdirs kept, outside path is an error, multiple matches is an error); `EMBED`/`RERANK` `auto` skips and `on` refuses; a non-loopback `HOST` warns |
| `test_services_run_args.py` | golden argv per role and backend: publish on `HOST` only; never `--network host`; secrets and volumes; GPU flags; NGL; hash stable and changing; **no key or token in argv or env** |
| `test_services_images.py` | the 5-step detection (fake CDI, `/dev/kfd`, `/dev/dri`, `nvidia-smi` with CUDA 12 or 13); arm64 without ROCm; override |
| `test_services_secrets.py` | one path inside the container, local and compose; no `--secret` without a token; a fingerprint change replaces the secret and recreates the container; no change touches nothing |
| `test_services_images_identity.py` | a tag-only match (`luna-llama:cuda13`) isn't reused; an unlabelled `luna-*` container gets the migration prompt; a busy port names the holder and skips the role |
| `test_services_lifecycle.py` | keep or replace by hash; `wait_ready` polls and times out (clock patched); `down` stops only what was started; `reset` asks before removing `luna-models` |
| `test_services_hf_token.py` | lookup order; the token is never written; a missing token is fine |
| `test_services_bench.py` | GPU, CPU or off, from mocked VRAM and latency |
| `test_installer_options.py` | the section 7.2 matrix, every flag against every other. Includes the old branch's breakages: `--voice url:` parses; `--backend llama` reaches the host build; no flags plus `--yes` gives main's choices; an explicit `--voice install` gets no container offer; `--gpu cpu` is honoured; no Podman doesn't crash; apt-get gets correct python-dev names |
| `test_installer_remote.py` | address validation; URLs for IPv4, IPv6 and hostnames; `llm.autostart = false`; key file is 0600 and asks before overwriting; the 200/401/no-answer messages; printed firewall rules |
| `test_installer_voice_choice.py` | the rules in 7.3 |
| `test_launch_autostart.py` | no `services` key: same calls in the same order as main; with autostart, up, then wait, then main's flow, then down; a chat container still loading means `start_llama()` isn't called; a container voice means `start_voice()` isn't called; `stop_on_exit=false` means no down |
| `test_compose_drift.py` | `compose.yaml` matches `run_args()`. If PyYAML isn't in CI, use a minimal parser; ask before touching CI |
| `test_main_unchanged.py` | in a **subprocess on a `git archive` copy with no `config.json`** (`config` reads the user's `config.json` at import), `LLAMA_URL`, `EMBED_URL`, `RERANK_URL`, `LLM_HEADERS` and `EMBED_HEADERS` match main's defaults; `EXTRAS`, the `mcp` extra and `mcpclient` config handling are unchanged |
| `test_doc_links.py` | section 11 checks |

### 12.2 Container tests (real Podman, opt-in)

- They live in `tests/containers/`, guarded by `unittest.skipUnless` on
  `LUNA_CONTAINER_TESTS` and `shutil.which("podman")`. CI skips them.
- They use dedicated `luna-test-*` containers, volumes and secrets on ports
  18080–18082 and 18899, so they never touch the user's real containers,
  models or key. Teardown always removes everything they created.
- Models: tiny public CPU GGUFs (chat, embed, rerank) are downloaded once
  into `luna-test-models`, and recorded in `tests/containers/models.env`.
- `LUNA_CONTAINER_TESTS=full` adds a small reasoning-capable chat model and
  a tiny vision model with an mmproj, for the reasoning and vision parity
  rows.

| File | Covers |
|---|---|
| `test_image_build.py` | cpu llama and TTS images build; entrypoint is shellcheck clean; runs rootless; the 5.3 checks |
| `test_llama_roles.py` | each role answers `/health`; 401 without the key, 200 with it; chat completes; embed returns the expected size; rerank orders correctly |
| `test_parity.py` | the section 10 table |
| `test_isolation.py` | `podman port` shows only `127.0.0.1`; `podman inspect` has no key or token; repo `git status` unchanged after up and down; nothing written outside Podman storage and temp |
| `test_lifecycle_live.py` | a second `up` does nothing; a setting edit recreates only that role; stopping embed leaves chat answering; `down` and `reset` leave nothing behind |
| `test_app_integration.py` | in a subprocess on a throwaway copy of the repo (`git archive`), with a test `config.json` and key: `_pick_backend()` gives llama; `embedmem.available()` and `reranker_available()` are true; recall returns ranked facts; TTS returns audio. The user's own config and key are never read |
| `test_compose_live.py` | skipped without compose: `up -d` with `LUNA_BIND=127.0.0.1` on test ports gives 4 healthy services; client config from the remote installer (loopback allowed only behind a test-only flag) passes the probe; `down -v` cleans up |
| `test_gpu.py` | skipped without CDI or `/dev/kfd`: the GPU image starts; `/props` shows GPU offload; `bench` recommends |

Run: `LUNA_CONTAINER_TESTS=1 python -m unittest discover -v tests/containers`

A CI job is optional and needs approval first. It would be a manual
`workflow_dispatch`, cpu only, on `ubuntu-24.04`. The existing `tests.yml`
stays untouched.

---

## 13. Definition of done

- `tests/check_scope_fence.py` passes.
- The existing tests pass unmodified.
- Every unit test in section 12.1 passes in CI.
- Every container test in section 12.2 passes locally. GPU and compose
  tests pass where their prerequisites exist, and skip otherwise.
- Unmodified `./start.sh` works:
  - with no services configured, exactly as on main;
  - against local containers;
  - against a remote compose host.
- `services reset` leaves no `luna-*` resources behind.
- `readme-containers.md`, `docs/containers.md` and every section 11 edit
  are done, and the link and command checks pass.

---

## 14. Stop and ask when

- A required llama.cpp flag or env var isn't in the pinned image.
- An app file outside the fence would need to change.
- kokoro-reader can't take a key. In that case remote TTS on :8899 is
  protected only by the IP allow-list. Ask whether that's acceptable.
- llama.cpp's router mode, or llama-swap, could serve embedding and
  reranking from fewer containers.
- RamaLama would make a better manager.
- Before starting P9.
- Someone wants MCP servers in containers. That's out of scope, and MCP is
  unchanged. Today it works like this:
  - The `mcp` extra only runs `pip install mcp` (the client SDK).
  - `main.py:_start_mcp()` calls `mcpclient.start()` for each entry in
    `config.json` → `mcp.servers`:
    - with `command`: spawned on the host over stdio (npx or uvx);
    - with `url`: reached over streamable HTTP.
  - Plugins call `mcpclient.connect_server()` on demand.

  A containerised MCP server would need only a `url` entry. That's a
  separate decision.
- Widening Luna's host Python range (3.14).
- Containerising the introspection lab.

---

## 15. Lessons from `feat/python-314-container` (don't repeat these)

- It rewrote `start.sh`, `embedmem.py` and `launch.py` instead of adding
  next to them.
- It called an undefined `_container_mode()` in `embedmem.py`.
- It ran side servers with `podman exec`, found them by PID through
  `/proc/net/tcp`, and needed a PID-1 guard.
- It translated model paths with `basename`, which loses subfolders.
- It used a hand-written `.env` parser.
- It wrote a predictable API key (`abcd…`).
- It bound `0.0.0.0` on the host with `--network host`.
- It re-entered a non-reentrant lock (the idle-stop deadlock).
- It used `sleep` where a readiness check belonged.
- It left dead code after `return` (local side servers never started).
- In the installer:
  - `choices=` on `--voice` broke `url:`;
  - `--backend llama` was given a new meaning;
  - `--llama` silently overrode `--backend`;
  - `--yes` flipped the voice default;
  - `--gpu` was ignored for containers;
  - unguarded `podman` calls crashed without Podman;
  - the deleted `mm = …` line raised a `NameError` on apt-get;
  - `input()` didn't handle EOF;
  - captured `install.sh` output hid its own fix;
  - `.part.part` temp files.
- In the TTS Dockerfiles:
  - unquoted `>=` pins became shell redirects;
  - `"torch[cpu]"` pulled CUDA torch;
  - the `nvidia/cuda` base doubled CUDA (11.4 GB);
  - the clone was unpinned;
  - there was no model cache volume;
  - the voice was hard-coded;
  - it used `--gpus all`.
- `server.sh`'s `exec llama-server` depended on `PATH`; the binary is
  `/app/llama-server`.
- Images were matched by tag, which would reuse a bare retag.

None of that branch's code is to be reused.
