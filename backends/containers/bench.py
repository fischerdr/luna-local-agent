"""python -m backends.containers bench: what goes on the GPU, and what on the CPU.

The GPU goes to chat and her voice first. The reranker, then the
embedder, get it only if there's room left after them (10% kept free);
otherwise they run on the CPU - the embedder is a few dozen tokens a
query; the reranker only when it finishes in time there, else it's off
and recall keeps the embedding order.

The CPU timings come from a short-lived luna-bench container on a spare
loopback port; nothing of yours is stopped or changed. The result is
printed, and written only if you say so: EMBED_NGL, RERANK_NGL and
TTS_DEVICE to backends/containers/services.local.env, RERANK to llama/server.local.env.
"""
import os
import re
import shutil
import socket
import statistics
import subprocess
import time

import requests

from backends.containers import CONTAINER, KEY_SECRET, LLAMA_DIR, VOLUME, images, keys, lifecycle, settings

SPARE = 0.10            # of the VRAM, kept free
RERANK_CPU_MAX = 1.5    # seconds for a 24-candidate rerank
CANDIDATES = 24
MIB = 1 << 20

QUERY = "what did I say about my sister's new job"
# 24 different facts, as recall sends: the same one 24 times would come out
# of llama.cpp's prompt cache and time almost nothing.
FACTS = [f"The user {what}, back in {month}." for what, month in zip((
    "said their sister started a new job as a nurse", "mentioned the night shifts are hard",
    "likes jazz records from the fifties", "has a cat called Miso who sleeps on the radiator",
    "is learning to bake sourdough bread", "drives a blue hatchback with a dented door",
    "plays chess online most evenings", "wants to visit Lisbon next spring",
    "keeps tomatoes and basil on the balcony", "fixed the kitchen tap themselves",
    "reads science fiction before bed", "runs five kilometres on Saturdays",
    "works on a home server with a big GPU", "dislikes loud restaurants",
    "has a brother who lives in Toronto", "collects old film cameras",
    "started a course in Spanish", "repainted the hallway green",
    "is allergic to peanuts", "watches football with their dad",
    "built a bookshelf out of pallets", "is saving for a new bike",
    "volunteers at the library on Thursdays", "prefers tea to coffee in the afternoon"),
    ("March", "April", "May", "June", "July", "August") * 4)]


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


def recommend(total, used, rerank_need, rerank_cpu, voice_need=0, embed_need=0):
    """What goes where, for these numbers: ({setting: value}, [why]).

    The GPU goes to chat and her voice first; the reranker, then the
    embedder, get it only if there's room after them. Otherwise they run on
    the CPU - the reranker only when a rerank there finishes in time
    (under RERANK_CPU_MAX), else it's off. total and used are MiB, used counting chat but nothing else
    (used=None without a GPU); voice_need is the voice's MiB on the GPU (0
    when it isn't asking for it), rerank_need and embed_need the side
    models'; rerank_cpu the seconds a 24-candidate rerank takes on the CPU,
    or None when it couldn't be timed."""
    out, why = {"EMBED_NGL": 0, "RERANK_NGL": 0, "RERANK": "auto"}, []
    gpu = bool(total) and used is not None
    room = total * (1 - SPARE) - used if gpu else 0
    if gpu and room < 0:
        why.append(f"chat alone takes about {used} of {total} MiB - nothing else goes on the GPU; if it "
                   "still runs out, a smaller model (the installer's --model 4b) or a lower NGL")
    if voice_need:
        if room >= voice_need:
            room -= voice_need
            why.append(f"her voice on the GPU, beside chat ({voice_need} MiB)")
        else:
            out["TTS_DEVICE"] = "cpu"
            why.append("no room for her voice beside chat - it goes on the CPU (faster than real time there)")
    for key, name, need in (("RERANK_NGL", "reranker", rerank_need), ("EMBED_NGL", "embedder", embed_need)):
        if gpu and need and room >= need:
            room -= need
            out[key] = 99
            why.append(f"the {name} fits on the GPU too ({need} MiB)")
        elif key == "RERANK_NGL" and need and (rerank_cpu is None or rerank_cpu >= RERANK_CPU_MAX):
            # Too slow on the CPU to finish inside embedmem's timeout: every
            # rerank would fail over to the embedding order anyway.
            out["RERANK"] = "off"
            took = f"{rerank_cpu:.1f} s" if rerank_cpu is not None else "too long to time"
            why.append(("no room for the reranker on the GPU, " if gpu else "")
                       + f"and a rerank takes {took} on the CPU - off; recall keeps the embedding order")
        else:
            where = " - no room left on the GPU" if gpu else ""
            took = f" ({rerank_cpu:.2f} s a rerank)" if key == "RERANK_NGL" and rerank_cpu is not None else ""
            why.append(f"the {name} on the CPU{where}{took}")
    if gpu:
        why.append(f"GPU: about {int(total * (1 - SPARE) - room)} of {total} MiB in use, "
                   f"{int(SPARE * 100)}% kept free")
    return out, why


# Measured on an RTX 3070 (2026-10-22/23, nvidia-smi per process, matched
# to containers by cgroup): a 4B Q6_K chat at 8k context took 4840 MiB (its
# file, its context, and llama.cpp's compute buffers), Kokoro on CUDA 882,
# the 0.6B Q8 reranker at 2k context 1124.
VOICE_MIB = 950         # Kokoro on CUDA: torch's context, the model and its working memory
CHAT_OVERHEAD = 1150    # beyond the file and its context (ctx / 20): compute buffers and the rest
SIDE_OVERHEAD = 550     # an embedder's or reranker's, beyond its file


def model_mib(m, s):
    """A role's model size in MiB, as near as we can tell (0 unknown): the
    file under LUNA_MODELS_DIR, or what Hugging Face says an -hf one is."""
    if m.host_mount:
        root = os.path.realpath(os.path.expanduser(s["LUNA_MODELS_DIR"]))
        path = os.path.join(root, m.file[len(settings.HOST_MODELS) + 1:])
        return os.path.getsize(path) // MIB if os.path.exists(path) else 0
    if m.hf:
        # What Hugging Face says it is: the same whether it's downloaded or not.
        from backends.containers.setup import hf_size

        return (hf_size(m.hf) or 0) // MIB
    return 0                      # a file already imported into the volume: unknown


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
        out("  no cpu image yet - python -m backends.containers setup builds it")
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
        docs = FACTS[:CANDIDATES]
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
        out("  no llama/.api_key yet - python -m backends.containers setup first")
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
        out(f"  reranker: {why}")

    mem = vram()
    found = lifecycle.containers()
    running = {role: found.get(CONTAINER[role], {}).get("state") == "running" for role in CONTAINER}
    total, used = mem if mem else (None, None)
    backend, _ = images.image_for_host(s["LLAMA_BACKEND"])
    voice_gpu = lifecycle.voice_state().get("container") == CONTAINER["tts"] and \
        lifecycle.tts_device(s, backend) == "cuda"
    need = model_mib(rerank, s) + SIDE_OVERHEAD if rerank else 0
    embed_need = model_mib(embed, s) + SIDE_OVERHEAD if embed else 0
    if mem:
        if not running["chat"]:
            try:
                chat, _ = settings.resolve("chat", s)
                used += model_mib(chat, s) + int(s["CTX"] or 16384) // 20 + CHAT_OVERHEAD
                out(f"  (chat isn't running: {used} MiB counted for it, from its size and CTX)")
            except settings.SettingsError:
                pass
        # What's on the GPU now besides chat is placed again, so it isn't counted.
        if rerank and running["rerank"] and int(s["RERANK_NGL"] or 0) > 0:
            used -= need
        if running["tts"] and voice_gpu:
            used -= VOICE_MIB
        if running["embed"] and int(s["EMBED_NGL"] or 0) > 0:
            used -= embed_need
    cpu = cpu_timing("rerank", s, rerank, key, out) if rerank else None
    place, why = recommend(total, used, need, cpu, VOICE_MIB if voice_gpu else 0, embed_need)
    for line in why:
        out(f"  {line}")
    out("  " + " ".join(f"{k}={v}" for k, v in place.items()))
    if not a.ask("Write that down (backends/containers/services.local.env, and RERANK in llama/server.local.env)?"):
        return True
    from backends.containers.setup import LOCAL_ENV, set_env

    for k in ("EMBED_NGL", "RERANK_NGL", "TTS_DEVICE"):
        if k in place:
            set_env(LOCAL_ENV, k, str(place[k]))
    # RERANK is llama/server.env's own switch: off goes in its local file,
    # and a reranker that runs again turns it back on there.
    local = os.path.join(LLAMA_DIR, "server.local.env")
    if place["RERANK"] == "off" or s["RERANK"] == "off":
        set_env(local, "RERANK", place["RERANK"])
    out("  written - python -m backends.containers up applies it")
    return True
