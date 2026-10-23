"""Luna's backend services in rootless Podman containers - optional.

    python -m backends.containers setup|up|down|status|logs <role>|bench|reset

The same four servers llama/start.sh and the installer's voice set up on
the host - chat, embeddings, reranker, Kokoro - each in its own
container, answering on the same addresses with the same key
(llama/.api_key). Luna herself doesn't change, or know: her config.json
URLs reach a container as they'd reach llama/start.sh.

Linux only. Nothing here is imported unless you ask for containers, so
Windows, macOS and machines without Podman never load it.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
LLAMA_DIR = os.path.join(ROOT, "llama")
KEY_FILE = os.path.join(LLAMA_DIR, ".api_key")      # main's LLAMA_KEY_FILE

LLAMA_ROLES = ("chat", "embed", "rerank")
ROLES = LLAMA_ROLES + ("tts",)

# One container per role, and its volumes and secrets. Every name starts
# with luna-/luna_, which is all reset ever removes.
CONTAINER = {"chat": "luna-llama", "embed": "luna-embed", "rerank": "luna-rerank", "tts": "luna-tts"}
VOLUME = {"models": "luna-models", "data": "luna-data", "tts": "luna-tts-cache"}
VOLUMES = tuple(VOLUME.values())
KEY_SECRET = "luna_api_key"
TOKEN_SECRET = "luna_hf_token"

# Rootless Podman this new or newer: secret create --replace is 4.7, CDI
# devices 4.1.
MIN_PODMAN = (4, 7, 0)
