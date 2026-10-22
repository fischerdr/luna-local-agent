"""deploy/compose.yaml for real: the four services up on loopback test
ports, healthy, and a client's probe (the installer's) finding them.
Skipped without podman-compose. Project luna-test, volumes luna-test-*.
"""
import importlib.util
import os
import shutil
import subprocess
import tempfile
import unittest
import urllib.request

from tests.containers import support
from tests.containers.support import live

DEPLOY = os.path.join(support.ROOT, "deploy")
COMPOSE = shutil.which("podman-compose")
# tests/offline.py stops urlopen for every test; an opener of its own
# reaches these servers, which are ours.
OPEN = urllib.request.build_opener().open
TEST_PORTS = (("chat", 18080, "/v1/models"), ("embeddings", 18081, "/v1/models"),
              ("reranker", 18082, "/v1/models"), ("voice", 18899, "/health"))


def installer():
    spec = importlib.util.spec_from_file_location("luna_installer_live",
                                                  os.path.join(support.ROOT, "installer", "install.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@live
@unittest.skipUnless(COMPOSE, "podman-compose isn't installed")
class ComposeLive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.key = support.secrets.token_urlsafe(32)
        key_file = os.path.join(cls.tmp.name, "api_key")
        token_file = os.path.join(cls.tmp.name, "hf_token")
        with open(key_file, "w") as f:
            f.write(cls.key)
        open(token_file, "w").close()                # empty: no token, the placeholder setup --server makes
        m = support.models()
        cls.env = dict(os.environ, COMPOSE_PROJECT_NAME="luna-test", LUNA_VOLUMES="luna-test",
                       LUNA_BIND="127.0.0.1", CHAT_PORT="18080", EMBED_PORT="18081", RERANK_PORT="18082",
                       TTS_PORT="18899", MODEL_HF=m["TEST_CHAT_HF"], EMBED_HF=m["TEST_EMBED_HF"],
                       RERANK_HF=m["TEST_RERANK_HF"], CTX="4096", PARALLEL="1", NGL="0", RERANK_NGL="0",
                       LUNA_API_KEY_FILE=key_file, HF_TOKEN_FILE=token_file)
        up = cls.compose("up", "-d", "--build")
        if up.returncode:
            cls.tearDownClass()
            raise AssertionError(f"compose up failed:\n{up.stdout[-2000:]}\n{up.stderr[-2000:]}")
        try:
            for role in ("chat", "embed", "rerank", "tts"):
                cls.ready(role)
        except Exception:
            cls.tearDownClass()
            raise

    @classmethod
    def compose(cls, *args):
        return subprocess.run([COMPOSE, "-f", "compose.yaml", *args], cwd=DEPLOY, env=cls.env,
                              capture_output=True, text=True, timeout=1800)

    @classmethod
    def ready(cls, role):
        import time

        import requests

        end = time.monotonic() + support.READY
        while time.monotonic() < end:
            try:
                if requests.get(support.url(role, "/health"), timeout=2).status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(1)
        raise AssertionError(f"compose {role} never became ready")

    @classmethod
    def tearDownClass(cls):
        cls.compose("down", "-v", "-t", "5")
        for vol in ("luna-test-data",):
            support.podman("volume", "rm", "-f", vol, check=False)
        cls.tmp.cleanup()

    def test_four_healthy_services_on_loopback(self):
        out = support.podman("ps", "--filter", "label=io.podman.compose.project=luna-test",
                             "--format", "{{.Names}} {{.Ports}}").stdout
        lines = [ln for ln in out.splitlines() if ln.strip()]
        self.assertEqual(len(lines), 4, out)
        for line in lines:
            self.assertIn("127.0.0.1:", line)
            self.assertNotIn("0.0.0.0", line)

    def test_the_clients_probe_finds_them(self):
        lines = installer().remote_probe("127.0.0.1", self.key, get=OPEN, ports=TEST_PORTS)
        self.assertEqual([good for _n, good, _l in lines], [True] * 4, lines)

    def test_a_wrong_key_is_named(self):
        lines = installer().remote_probe("127.0.0.1", "not-the-key", get=OPEN, ports=TEST_PORTS)
        self.assertIn("key doesn't match", lines[0][2])

    def test_empty_token_file_is_no_token(self):
        name = support.podman("ps", "--filter", "label=io.podman.compose.project=luna-test",
                              "--filter", "label=io.podman.compose.service=embed",
                              "--format", "{{.Names}}").stdout.strip()
        env = support.podman("exec", name, "sh", "-c", "tr '\\0' '\\n' < /proc/1/environ").stdout
        self.assertNotIn("HF_TOKEN", env)


if __name__ == "__main__":
    unittest.main()
