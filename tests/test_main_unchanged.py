"""Main's defaults are still main's: the model, embedding and reranker URLs
and headers, the installer's extras (the mcp one too), and how MCP servers
are configured - on this branch and on the commit it started from, each
read in a subprocess on a git archive copy with no config.json and a HOME
of its own (config reads config.json when it's imported).

    python -m unittest discover tests        (from the repo folder)
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.offline import FAKE  # noqa: E402,F401  (first: no real servers)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROBE = r'''
import hashlib, importlib.util, json, os, sys
sys.path.insert(0, os.getcwd())
import config
spec = importlib.util.spec_from_file_location("inst", os.path.join("installer", "install.py"))
inst = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inst)
with open("mcpclient.py", "rb") as f:
    mcp = hashlib.sha256(f.read()).hexdigest()
print("PROBE" + json.dumps({
    "LLAMA_URL": config.LLAMA_URL, "EMBED_URL": config.EMBED_URL, "RERANK_URL": config.RERANK_URL,
    "LLM_HEADERS": getattr(config, "LLM_HEADERS", None), "EMBED_HEADERS": getattr(config, "EMBED_HEADERS", None),
    "EXTRAS": {k: v[0] for k, v in inst.EXTRAS.items()}, "mcpclient": mcp,
    "MCP": {k: getattr(config, k) for k in dir(config) if k.startswith("MCP")},
}, default=str))
'''


def probe(rev):
    with tempfile.TemporaryDirectory() as tmp:
        repo, home = os.path.join(tmp, "repo"), os.path.join(tmp, "home")
        os.makedirs(repo)
        os.makedirs(home)
        archive = subprocess.run(["git", "-C", ROOT, "archive", rev], capture_output=True)
        if archive.returncode:
            return None
        subprocess.run(["tar", "-x", "-C", repo], input=archive.stdout, check=True)
        for leftover in ("config.json", os.path.join("llama", ".api_key")):
            if os.path.exists(os.path.join(repo, leftover)):
                os.remove(os.path.join(repo, leftover))
        env = {"HOME": home, "PATH": os.environ.get("PATH", "/usr/bin:/bin")}
        # The probe's own copy of tests/offline.py keeps it off the network.
        code = "import tests.offline\n" + PROBE
        r = subprocess.run([sys.executable, "-c", code], cwd=repo, env=env, capture_output=True, text=True,
                           timeout=120)
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("PROBE")]
    if not line:
        raise AssertionError(f"probe failed on {rev}:\n{r.stderr[-2000:]}")
    return json.loads(line[0][len("PROBE"):])


class MainUnchanged(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = subprocess.run(["git", "-C", ROOT, "merge-base", "HEAD", "main"], capture_output=True, text=True)
        rev = os.environ.get("FENCE_BASE") or base.stdout.strip()
        cls.main = probe(rev) if rev else None
        if cls.main is None:
            raise unittest.SkipTest("main isn't in this checkout's history")
        cls.here = probe("HEAD")

    def test_urls_and_headers(self):
        for key in ("LLAMA_URL", "EMBED_URL", "RERANK_URL", "LLM_HEADERS", "EMBED_HEADERS"):
            with self.subTest(key=key):
                self.assertEqual(self.here[key], self.main[key])

    def test_the_defaults_are_the_documented_ones(self):
        self.assertEqual(self.here["LLAMA_URL"], "http://127.0.0.1:8080/v1/chat/completions")
        self.assertEqual(self.here["EMBED_URL"], "http://127.0.0.1:8081/v1/embeddings")
        self.assertEqual(self.here["RERANK_URL"], "http://127.0.0.1:8082/v1/rerank")

    def test_extras_and_mcp(self):
        self.assertEqual(self.here["EXTRAS"], self.main["EXTRAS"])
        self.assertEqual(self.here["EXTRAS"]["mcp"], ["mcp"])
        self.assertEqual(self.here["mcpclient"], self.main["mcpclient"])
        self.assertEqual(self.here["MCP"], self.main["MCP"])


if __name__ == "__main__":
    unittest.main()
