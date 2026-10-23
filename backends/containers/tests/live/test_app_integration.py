"""Luna's own code, unmodified, against the containers.

It runs in a subprocess, on a throwaway copy of the repo (git archive),
with a test config.json and the test key - and a HOME of its own, so your
config.json, llama/.api_key and ~/.config/ai-voice are never read.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

from backends.containers.tests.live import support
from backends.containers.tests.live.support import live, url

ROLES = ("chat", "embed", "rerank")

PROBE = r'''
import json
import config
import embedmem
import speech

facts = ["The user likes jazz.", "The user has a cat named Miso.",
         "Paris is in France.", "The user drives a blue car."]
query = "what pet does the user have"
out = {"backend": config.LLM_BACKEND, "embed": embedmem.available(),
       "rerank": embedmem.reranker_available()}
ranked = embedmem.rank(query, facts)
out["rank"] = [f for _, f in ranked] if ranked else None
out["rerank_order"] = embedmem.rerank(query, facts)
samples, rate = speech._synthesize("Hello from the container.")
out["tts"] = [len(samples), rate]
print("PROBE" + json.dumps(out))
'''


@live
class UnmodifiedLuna(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        m = support.models()
        key = support.make_key()
        try:
            support.start("chat", m["TEST_CHAT_HF"], {"CTX": "4096", "PARALLEL": "1", "NGL": "0"})
            support.start("embed", m["TEST_EMBED_HF"])
            support.start("rerank", m["TEST_RERANK_HF"], {"RERANK_NGL": "0"})
            support.start_tts()
            for role in ROLES + ("tts",):
                support.wait_ready(role)
            cls.out = cls.probe(key)
        finally:
            support.cleanup(*ROLES, "tts")

    @classmethod
    def probe(cls, key):
        with tempfile.TemporaryDirectory() as tmp:
            repo = os.path.join(tmp, "repo")
            os.makedirs(os.path.join(tmp, "home"))
            os.makedirs(repo)
            archive = subprocess.run(["git", "-C", support.ROOT, "archive", "HEAD"],
                                     check=True, capture_output=True).stdout
            subprocess.run(["tar", "-x", "-C", repo], input=archive, check=True)
            cfg = {"llm": {"backend": "auto",
                           "llama_url": url("chat", "/v1/chat/completions"),
                           "embed_url": url("embed", "/v1/embeddings"),
                           "rerank_url": url("rerank", "/v1/rerank")},
                   "tts": {"url": url("tts")}, "voice": "af_bella"}
            with open(os.path.join(repo, "config.json"), "w") as f:
                json.dump(cfg, f)
            with open(os.path.join(repo, "llama", ".api_key"), "w") as f:
                f.write(key)
            os.chmod(os.path.join(repo, "llama", ".api_key"), 0o600)
            env = {"HOME": os.path.join(tmp, "home"), "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                   "XDG_RUNTIME_DIR": os.environ.get("XDG_RUNTIME_DIR", tmp)}
            r = subprocess.run([sys.executable, "-c", PROBE], cwd=repo, env=env,
                               capture_output=True, text=True, timeout=300)
        line = [ln for ln in r.stdout.splitlines() if ln.startswith("PROBE")]
        if not line:
            raise AssertionError(f"the probe failed:\n{r.stdout[-2000:]}\n{r.stderr[-3000:]}")
        return json.loads(line[0][len("PROBE"):])

    def test_auto_picks_llama(self):
        self.assertEqual(self.out["backend"], "llama")

    def test_embeddings_and_reranker_found(self):
        self.assertTrue(self.out["embed"])
        self.assertTrue(self.out["rerank"])

    def test_recall_ranks_the_answer_first(self):
        self.assertEqual(self.out["rank"][0], "The user has a cat named Miso.")
        self.assertEqual(self.out["rerank_order"][0], "The user has a cat named Miso.")

    def test_she_speaks(self):
        samples, rate = self.out["tts"]
        self.assertEqual(rate, 24000)
        self.assertGreater(samples, rate // 2)


if __name__ == "__main__":
    unittest.main()
