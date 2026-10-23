"""What the containers keep to themselves: the key and the Hugging Face
token reach llama-server and nothing else, and the repo is untouched.

A gated model with your own token is opt-in, since it's your token and
your accepted licence: LUNA_TEST_GATED=<user>/<repo>[:quant] (the token
from HF_TOKEN or huggingface-cli login).
"""
import os
import subprocess
import tempfile
import unittest

from backends.containers import keys, lifecycle
from backends.containers.tests.live import support
from backends.containers.tests.live.support import live, podman

FAKE_TOKEN = "hf_luna_test_not_a_real_token_0123456789"
GATED = "google/gemma-3-1b-it-qat-q4_0-gguf"
TOKEN_SECRET = "luna_test_hf_token"


def git_status():
    return subprocess.run(["git", "-C", support.ROOT, "status", "--porcelain"],
                          capture_output=True, text=True).stdout


@live
class Secrets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before = git_status()
        cls.key = support.make_key()
        podman("secret", "create", "--replace", TOKEN_SECRET, "-", input=FAKE_TOKEN)
        try:
            podman("run", "-d", "--replace", "--name", "luna-test-embed",
                   "-p", f"127.0.0.1:{support.PORTS['embed']}:8080", "-v", f"{support.MODELS}:/models",
                   "--secret", f"{support.KEY_SECRET},target=api_key",
                   "--secret", f"{TOKEN_SECRET},target=hf_token",
                   "-e", "ROLE=embed", "-e", f"LUNA_HF={support.models()['TEST_EMBED_HF']}",
                   support.build_llama())
            support.wait_ready("embed")
        except Exception:
            cls.tearDownClass()
            raise

    @classmethod
    def tearDownClass(cls):
        support.cleanup("embed")
        podman("secret", "rm", TOKEN_SECRET, check=False)

    def test_token_reaches_llama_server(self):
        env = podman("exec", "luna-test-embed", "sh", "-c", "tr '\\0' '\\n' < /proc/1/environ").stdout
        self.assertIn(f"HF_TOKEN={FAKE_TOKEN}", env)

    def test_neither_secret_in_inspect_or_logs(self):
        inspect = podman("inspect", "luna-test-embed").stdout
        logs = podman("logs", "luna-test-embed").stdout + podman("logs", "luna-test-embed").stderr
        for secret in (self.key, FAKE_TOKEN):
            self.assertNotIn(secret, inspect)
            self.assertNotIn(secret, logs)

    def test_neither_secret_in_argv(self):
        cmdline = podman("exec", "luna-test-embed", "sh", "-c", "tr '\\0' ' ' < /proc/1/cmdline").stdout
        self.assertIn("--api-key-file /run/secrets/api_key", cmdline)
        self.assertNotIn(self.key, cmdline)
        self.assertNotIn(FAKE_TOKEN, cmdline)

    def test_repo_untouched(self):
        self.assertEqual(git_status(), self.before)


@live
class EmptyToken(unittest.TestCase):
    """Compose mounts an empty file when there's no token: that means none."""

    def test_not_exported(self):
        # Podman won't make an empty secret; compose's file secrets are
        # mounted files, so an empty one is mounted here the same way.
        support.make_key()
        empty = tempfile.NamedTemporaryFile(prefix="luna-test-token-")
        self.addCleanup(empty.close)
        try:
            podman("run", "-d", "--replace", "--name", "luna-test-embed", "--security-opt", "label=disable",
                   "-p", f"127.0.0.1:{support.PORTS['embed']}:8080", "-v", f"{support.MODELS}:/models",
                   "--secret", f"{support.KEY_SECRET},target=api_key",
                   "-v", f"{empty.name}:/run/secrets/hf_token:ro",
                   "-e", "ROLE=embed", "-e", f"LUNA_HF={support.models()['TEST_EMBED_HF']}",
                   support.build_llama())
            support.wait_ready("embed")
            env = podman("exec", "luna-test-embed", "sh", "-c", "tr '\\0' '\\n' < /proc/1/environ").stdout
        finally:
            support.cleanup("embed")
        self.assertNotIn("HF_TOKEN", env)


@live
class Gated(unittest.TestCase):
    def test_without_a_token_it_fails_with_the_fix(self):
        support.make_key()
        try:
            r = podman("run", "--rm", "-v", f"{support.MODELS}:/models",
                       "--secret", f"{support.KEY_SECRET},target=api_key",
                       "-e", "ROLE=embed", "-e", f"LUNA_HF={GATED}", support.build_llama(),
                       check=False, timeout=120)
        finally:
            podman("secret", "rm", support.KEY_SECRET, check=False)
        hint = lifecycle.trouble("embed", log=r.stdout + r.stderr)
        self.assertIsNotNone(hint, r.stderr[-1500:])
        self.assertIn("gated", hint)
        self.assertIn("huggingface-cli login", hint)

    @unittest.skipUnless(os.environ.get("LUNA_TEST_GATED"), "set LUNA_TEST_GATED=<repo> to use your own token")
    def test_with_your_token_it_downloads(self):
        token, _ = keys.token_source()
        self.assertTrue(token, "no Hugging Face token found (HF_TOKEN or huggingface-cli login)")
        podman("secret", "create", "--replace", TOKEN_SECRET, "-", input=token)
        support.make_key()
        try:
            podman("run", "-d", "--replace", "--name", "luna-test-embed",
                   "-p", f"127.0.0.1:{support.PORTS['embed']}:8080", "-v", f"{support.MODELS}:/models",
                   "--secret", f"{support.KEY_SECRET},target=api_key",
                   "--secret", f"{TOKEN_SECRET},target=hf_token",
                   "-e", "ROLE=chat", "-e", "CTX=2048", "-e", "NGL=0", "-e", "PARALLEL=1",
                   "-e", f"LUNA_HF={os.environ['LUNA_TEST_GATED']}", support.build_llama())
            support.wait_ready("embed")
            for out in (podman("inspect", "luna-test-embed").stdout, podman("logs", "luna-test-embed").stderr):
                self.assertNotIn(token, out)
        finally:
            support.cleanup("embed")
            podman("secret", "rm", TOKEN_SECRET, check=False)


if __name__ == "__main__":
    unittest.main()
