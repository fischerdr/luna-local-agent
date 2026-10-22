"""The API key and the Hugging Face token, as Podman secrets.

Inside every container they're files: /run/secrets/api_key and, when
there is a token, /run/secrets/hf_token. Never in argv, an image, an
env-file or a log.

The key is llama/.api_key - the same file Luna and llama/start.sh use,
made by start.sh's own recipe when it's missing. The token is yours:
found where Hugging Face keeps it, read, never copied anywhere but into
its secret.

Podman copies a secret into a container when the container is made, so
a changed key or token means new containers: each secret carries its
fingerprint (a SHA-256, never the value) as a label, and so does each
container's config hash.
"""
import base64
import hashlib
import os
import subprocess

from services import KEY_FILE, KEY_SECRET, TOKEN_SECRET

FINGERPRINT = "luna.fingerprint"


def fingerprint(value):
    """The SHA-256 of a secret: safe to keep and compare, unlike it."""
    return hashlib.sha256(value.encode()).hexdigest() if value else ""


def read_key(path=KEY_FILE):
    """llama/.api_key's key, or "" when there isn't one."""
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return ""


def make_key(path=KEY_FILE):
    """start.sh's recipe: 32 random bytes, base64 without /+=, readable
    only by you. Returns the key."""
    key = base64.b64encode(os.urandom(32)).decode().translate({ord(c): None for c in "/+=\n"})
    old = os.umask(0o077)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(key)
    finally:
        os.umask(old)
    os.chmod(path, 0o600)
    return key


def token_source(env=None, home=None):
    """(token, where it came from), in Hugging Face's own order: HF_TOKEN,
    then $HF_HOME/token, then ~/.cache/huggingface/token. ("", "") when
    there's none - public models don't need one."""
    env = os.environ if env is None else env
    home = home or os.path.expanduser("~")
    if env.get("HF_TOKEN", "").strip():
        return env["HF_TOKEN"].strip(), "HF_TOKEN"
    paths = []
    if env.get("HF_HOME"):
        paths.append(os.path.join(env["HF_HOME"], "token"))
    paths.append(os.path.join(home, ".cache", "huggingface", "token"))
    for path in paths:
        try:
            with open(path, encoding="utf-8") as f:
                token = f.read().strip()
        except OSError:
            continue
        if token:
            return token, path
    return "", ""


def secret_fingerprint(name, podman=subprocess.run):
    """The fingerprint label on a Podman secret, None if there's no secret."""
    r = podman(["podman", "secret", "inspect", "--format", "{{json .Spec.Labels}}", name],
               capture_output=True, text=True)
    if r.returncode:
        return None
    import json

    return (json.loads(r.stdout or "{}") or {}).get(FINGERPRINT, "")


def ensure_secret(name, value, podman=subprocess.run):
    """Make or replace a secret, but only when its value changed. The value
    goes in on stdin. True if it changed."""
    fp = fingerprint(value)
    if secret_fingerprint(name, podman) == fp:
        return False
    podman(["podman", "secret", "create", "--replace", "--label", f"{FINGERPRINT}={fp}", name, "-"],
           input=value, capture_output=True, text=True, check=True)
    return True


def remove_secret(name, podman=subprocess.run):
    """Remove a secret if it's there (a token you've since logged out of)."""
    podman(["podman", "secret", "rm", name], capture_output=True, text=True)


def sync(key, token, podman=subprocess.run):
    """Both secrets, current. Returns the names that changed."""
    changed = []
    if ensure_secret(KEY_SECRET, key, podman):
        changed.append(KEY_SECRET)
    if token:
        if ensure_secret(TOKEN_SECRET, token, podman):
            changed.append(TOKEN_SECRET)
    elif secret_fingerprint(TOKEN_SECRET, podman) is not None:
        remove_secret(TOKEN_SECRET, podman)
        changed.append(TOKEN_SECRET)
    return changed
