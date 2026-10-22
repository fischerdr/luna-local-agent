"""python -m services bench: where the side models go on a small GPU.

The embedder always runs on the CPU (EMBED_NGL=0): a query is a few
dozen tokens and fact vectors are cached, so it costs no VRAM and stays
well inside embedmem's 2-second timeout. The reranker:

- on the GPU, when chat and the reranker fit with about 10% to spare;
- else on the CPU, when a 24-candidate rerank (embedmem's
  RERANK_CANDIDATES) takes under about 1.5 seconds there;
- else off (RERANK=off), as on a Mac - recall keeps the embedding order.

The CPU timings come from a short-lived luna-bench container on a spare
loopback port; nothing of yours is stopped or changed. The result is
printed, and written only if you say so: EMBED_NGL and RERANK_NGL to
services/services.local.env, RERANK=off to llama/server.local.env.
"""
import os
import re
import shutil
import socket
import statistics
import subprocess
import time

import requests

from services import CONTAINER, KEY_SECRET, LLAMA_DIR, VOLUME, images, keys, lifecycle, settings

SPARE = 0.10            # of the VRAM, kept free
RERANK_CPU_MAX = 1.5    # seconds for a 24-candidate rerank
CANDIDATES = 24
MIB = 1 << 20

QUERY = "what did I say about my sister's new job"
FACT = ("The user's sister started a new job as a nurse at the hospital in March, "
        "and the user mentioned the night shifts are hard on her.")


def vram(run=subprocess.run):
    """(total, used) MiB on the first GPU, or None without one."""
    if shutil.which("nvidia-smi"):
        out = run(["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"],
                  capture_output=True, text=True).stdout
        nums = re.findall(r"\d+", out.splitlines()[0] if out.strip() else "")
        if len(nums) >= 2:
            return int(nums[0]), int(nums[1])
    if shutil.which("rocm-smi"):
        out = run(["rocm-smi", "--showmeminfo", "vram", "--csv"], capture_output=True, text=True).stdout
        nums = [int(n) for n in re.findall(r"\b\d{6,}\b", out)]
        if len(nums) >= 2:
            return nums[0] // MIB, nums[1] // MIB
    return None


def recommend(total, used, rerank_need, rerank_cpu):
    """(EMBED_NGL, RERANK_NGL, RERANK) for these numbers, and why.

    total and used are MiB with chat loaded (used=None without a GPU);
    rerank_need is the reranker's MiB on the GPU; rerank_cpu the seconds a
    24-candidate rerank takes on the CPU (None if it couldn't be timed)."""
    if total and used is not None and used + rerank_need <= total * (1 - SPARE):
        return (0, 99, "auto"), (f"chat and the reranker fit: {used + rerank_need} of {total} MiB, "
                                 f"{int(SPARE * 100)}% to spare - reranker on the GPU")
    tight = f"no room on the GPU ({used} + {rerank_need} of {total} MiB)" if total else "no GPU"
    if rerank_cpu is not None and rerank_cpu < RERANK_CPU_MAX:
        return (0, 0, "auto"), f"{tight}; a rerank takes {rerank_cpu:.2f} s on the CPU - reranker on the CPU"
    took = f"{rerank_cpu:.2f} s" if rerank_cpu is not None else "too long to time"
    return (0, 0, "off"), (f"{tight}, and a rerank on the CPU takes {took} - reranker off; recall keeps "
                           "the embedding order")


def model_mib(m, s):
    """A role's model size in MiB, as near as we can tell (0 unknown)."""
    if m.host_mount:
        root = os.path.realpath(os.path.expanduser(s["LUNA_MODELS_DIR"]))
        path = os.path.join(root, m.file[len(settings.HOST_MODELS) + 1:])
        return os.path.getsize(path) // MIB if os.path.exists(path) else 0
    found = lifecycle.podman("run", "--rm", "--entrypoint", "sh", "-v", f"{VOLUME['models']}:/models:ro",
                             images.find("cpu") or "", "-c",
                             f"du -Lbc $(find /models -iname '*{m.hf.split('/')[-1].split(':')[0]}*' "
                             "-name '*.gguf' ! -iname '*mmproj*' 2>/dev/null) 2>/dev/null | tail -1")
    nums = re.findall(r"^\d+", found.stdout.strip())
    return int(nums[0]) // MIB if nums else 0


def spare_port():
    """A free TCP port on loopback, for a short-lived bench container."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def timed(post, url, body, key, runs=3):
    """The median seconds of a few requests (the first warms it up)."""
    headers = {"Authorization": f"Bearer {key}"}
    post(url, json=body, headers=headers, timeout=60).raise_for_status()
    times = []
    for _ in range(runs):
        t = time.monotonic()
        post(url, json=body, headers=headers, timeout=60).raise_for_status()
        times.append(time.monotonic() - t)
    return statistics.median(times)


def cpu_timing(role, s, m, key, out=print):
    """Seconds for one embedding or a 24-candidate rerank on the CPU, in a
    short-lived container - or None."""
    image = images.find("cpu")
    if not image:
        out("  no cpu image yet - python -m services setup builds it")
        return None
    port = spare_port()
    name = f"luna-bench-{role}"
    env = settings.env_lines(role, dict(s, EMBED_NGL="0", RERANK_NGL="0"), m)
    args = ["run", "-d", "--replace", "--name", name, "-p", f"127.0.0.1:{port}:8080",
            "-v", f"{VOLUME['models']}:/models", "--secret", f"{KEY_SECRET},target=api_key"]
    if m.host_mount:
        args += ["-v", f"{os.path.realpath(os.path.expanduser(s['LUNA_MODELS_DIR']))}:{settings.HOST_MODELS}:ro",
                 "--security-opt", "label=disable"]
    for line in env.splitlines():
        args += ["-e", line]
    try:
        if lifecycle.podman(*args, image).returncode:
            return None
        url = f"http://127.0.0.1:{port}"
        if not lifecycle.wait_ready(url, 300, alive=lambda: lifecycle.podman(
                "inspect", "-f", "{{.State.Running}}", name).stdout.strip() == "true"):
            return None
        if role == "embed":
            return timed(requests.post, url + "/v1/embeddings", {"input": QUERY}, key)
        docs = [FACT] * CANDIDATES
        return timed(requests.post, url + "/v1/rerank", {"query": QUERY, "documents": docs}, key)
    except requests.RequestException:
        return None
    finally:
        lifecycle.podman("rm", "-f", "-t", "2", name)


def bench(a, out=print):
    """Measure, recommend, and (asked) write it. True when it could say."""
    s = settings.load()
    key = keys.read_key()
    if not key:
        out("  no llama/.api_key yet - python -m services setup first")
        return False
    keys.sync(key, keys.token_source()[0])

    try:
        embed, _ = settings.resolve("embed", s)
        rerank, why = settings.resolve("rerank", s)
    except settings.SettingsError as e:
        out(f"  {e}")
        return False
    if embed:
        took = cpu_timing("embed", s, embed, key, out)
        out(f"  embedder on the CPU: {took * 1000:.0f} ms a query" if took is not None
            else "  embedder: couldn't time it")
    if not rerank:
        out(f"  reranker: {why} - nothing to place")
        return True

    mem = vram()
    chat_up = lifecycle.containers().get(CONTAINER["chat"], {}).get("state") == "running"
    total, used = mem if mem else (None, None)
    if mem and not chat_up:
        try:
            chat, _ = settings.resolve("chat", s)
            used += model_mib(chat, s) + int(s["CTX"] or 16384) // 20 + 512
            out(f"  (chat isn't running: {used} MiB counted for it, from its size and CTX)")
        except settings.SettingsError:
            pass
    if mem and chat_up and lifecycle.containers().get(CONTAINER["rerank"], {}).get("state") == "running" \
            and int(s["RERANK_NGL"] or 0) > 0:
        used -= model_mib(rerank, s) + 400     # the reranker already on the GPU is counted in used
    need = model_mib(rerank, s) + 400
    cpu = cpu_timing("rerank", s, rerank, key, out)
    (embed_ngl, rerank_ngl, switch), reason = recommend(total, used, need, cpu)
    out(f"  {reason}")
    out(f"  EMBED_NGL={embed_ngl} RERANK_NGL={rerank_ngl}" + (" RERANK=off" if switch == "off" else ""))
    if not a.ask("Write that down (services/services.local.env, and llama/server.local.env for RERANK=off)?"):
        return True
    from services.setup import LOCAL_ENV, set_env

    set_env(LOCAL_ENV, "EMBED_NGL", str(embed_ngl))
    set_env(LOCAL_ENV, "RERANK_NGL", str(rerank_ngl))
    if switch == "off":
        set_env(os.path.join(LLAMA_DIR, "server.local.env"), "RERANK", "off")
    out("  written - python -m services up applies it")
    return True
