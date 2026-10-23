"""Shared by the container tests: real rootless Podman, opt-in.

    LUNA_CONTAINER_TESTS=1 python -m unittest discover -v -s backends/containers/tests/live -t .
    LUNA_CONTAINER_TESTS=full ...      (adds the reasoning and vision rows)

Everything is luna-test-*: its own containers, secret and image, on ports
18080-18082 and 18899, so your own containers, models and key are never
touched. Teardown removes all of it except the luna-test-models volume,
and luna-test-tts-cache, which keep the tiny test models (models.env)
and the voice model between runs.
"""
import os
import secrets
import shutil
import subprocess
import time
import unittest

from tests.offline import FAKE

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
HERE = os.path.dirname(os.path.abspath(__file__))

MODE = os.environ.get("LUNA_CONTAINER_TESTS", "")
LIVE = bool(MODE) and bool(shutil.which("podman"))
FULL = MODE == "full"
live = unittest.skipUnless(LIVE, "set LUNA_CONTAINER_TESTS=1 (and have podman) to run")
full = unittest.skipUnless(LIVE and FULL, "LUNA_CONTAINER_TESTS=full only")

PORTS = {"chat": 18080, "embed": 18081, "rerank": 18082, "tts": 18899}
LLAMA_IMAGE = "localhost/luna-test-llama:cpu"
TTS_IMAGE = "localhost/luna-test-tts:cpu"
TTS_CACHE = "luna-test-tts-cache"
# The Dockerfile's default (cpu) base; services/images.py will own the pins.
LLAMA_BASE_DIGEST = "sha256:8c74db29f1a2bef3f0a331b52c4d6214601dfceed4bee0b94cc48b3b23ff257d"
MODELS = "luna-test-models"
KEY_SECRET = "luna_test_api_key"

# The first run downloads the models; after that a role is up in seconds.
READY = int(os.environ.get("LUNA_TEST_READY", "900"))

# tests/offline.py keeps every test away from real servers; these are ours.
for _port in PORTS.values():
    if f"http://127.0.0.1:{_port}" not in FAKE:
        FAKE.append(f"http://127.0.0.1:{_port}")


def podman(*args, check=True, **kw):
    """Run podman, capturing its output as text."""
    return subprocess.run(["podman", *args], check=check, capture_output=True,
                          text=True, **kw)


def models():
    """models.env, read by bash as every other env file here is."""
    out = subprocess.run(["env", "-i", "bash", "-c", 'set -a; source "$0"; env -0',
                          os.path.join(HERE, "models.env")],
                         check=True, capture_output=True, text=True).stdout
    return {k: v for k, _, v in (kv.partition("=") for kv in out.split("\0") if kv)
            if k.startswith("TEST_")}


_built = {}


def build_llama():
    """The llama image from services/llama, built once per run."""
    if LLAMA_IMAGE not in _built:
        podman("build", "-q", "--label", "luna.image-version=test",
               "--label", f"luna.base-digest={LLAMA_BASE_DIGEST}", "--label", "luna.backend=server",
               "-t", LLAMA_IMAGE, os.path.join(ROOT, "backends", "containers", "llama"))
        _built[LLAMA_IMAGE] = True
    return LLAMA_IMAGE


def build_tts():
    """The voice image from services/tts (cpu), built once per run."""
    if TTS_IMAGE not in _built:
        podman("build", "-q", "--label", "luna.image-version=test", "-t", TTS_IMAGE,
               os.path.join(ROOT, "backends", "containers", "tts"))
        _built[TTS_IMAGE] = True
    return TTS_IMAGE


def start_tts(voice="af_bella"):
    """The voice as luna-test-tts on 127.0.0.1:18899, its model cached."""
    podman("run", "-d", "--replace", "--init", "--name", "luna-test-tts",
           "-p", f"127.0.0.1:{PORTS['tts']}:8899", "-v", f"{TTS_CACHE}:/cache",
           "-e", "KOKORO_DEVICE=cpu", "-e", f"KOKORO_VOICE={voice}", build_tts())
    return "luna-test-tts"


def make_key():
    """A fresh random key in the test secret; returns it for the requests."""
    key = secrets.token_urlsafe(32)
    podman("secret", "create", "--replace", KEY_SECRET, "-", input=key)
    return key


def url(role, path=""):
    """A role's test address on loopback."""
    return f"http://127.0.0.1:{PORTS[role]}{path}"


def start(role, hf, env=None, args=()):
    """One llama role as luna-test-<role>, published on loopback only."""
    name = f"luna-test-{role}"
    cmd = ["run", "-d", "--replace", "--name", name,
           "-p", f"127.0.0.1:{PORTS[role]}:8080",
           "-v", f"{MODELS}:/models",
           "--secret", f"{KEY_SECRET},target=api_key",
           "-e", f"ROLE={role}", "-e", f"LUNA_HF={hf}"]
    for k, v in (env or {}).items():
        cmd += ["-e", f"{k}={v}"]
    podman(*cmd, build_llama(), *args)
    return name


def wait_ready(role, timeout=READY):
    """Poll /health until it says ok; fail with the container's log."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            if requests.get(url(role, "/health"), timeout=2).status_code == 200:
                return
        except requests.RequestException:
            pass
        state = podman("inspect", "-f", "{{.State.Running}}", f"luna-test-{role}",
                       check=False).stdout.strip()
        if state != "true":
            break
        time.sleep(1)
    log = podman("logs", "--tail", "30", f"luna-test-{role}", check=False)
    raise AssertionError(f"{role} never became ready:\n{log.stdout}{log.stderr}")


def cleanup(*roles):
    """Remove our containers (and the key secret); the models stay."""
    for role in roles:
        podman("rm", "-f", "-t", "2", f"luna-test-{role}", check=False)
    podman("secret", "rm", KEY_SECRET, check=False)
