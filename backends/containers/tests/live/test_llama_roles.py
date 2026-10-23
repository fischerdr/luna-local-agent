"""Each llama role in its container answers as main's app expects.

The key, chat, embeddings and reranking (plans/container-migration-plan-v2.md,
12.2).
"""
import unittest

import requests

from backends.containers.tests.live import support
from backends.containers.tests.live.support import live, podman, url

ROLES = ("chat", "embed", "rerank")


@live
class LlamaRoles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        m = support.models()
        cls.key = support.make_key()
        try:
            # Small and quick on a cpu: a short context, no thinking.
            support.start("chat", m["TEST_CHAT_HF"],
                          {"CTX": "4096", "PARALLEL": "1", "REASONING_BUDGET": "0", "NGL": "0"})
            support.start("embed", m["TEST_EMBED_HF"])
            support.start("rerank", m["TEST_RERANK_HF"], {"RERANK_NGL": "0"})
            for role in ROLES:
                support.wait_ready(role)
        except Exception:
            support.cleanup(*ROLES)
            raise

    @classmethod
    def tearDownClass(cls):
        support.cleanup(*ROLES)

    def auth(self):
        return {"Authorization": f"Bearer {self.key}"}

    def test_health_answers(self):
        for role in ROLES:
            with self.subTest(role=role):
                self.assertEqual(requests.get(url(role, "/health"), timeout=5).status_code, 200)

    def test_key_is_required(self):
        for role, path in (("chat", "/v1/models"), ("embed", "/v1/models"), ("rerank", "/v1/models")):
            with self.subTest(role=role):
                self.assertEqual(requests.get(url(role, path), timeout=5).status_code, 401)
                self.assertEqual(requests.get(url(role, path), headers=self.auth(),
                                              timeout=5).status_code, 200)

    def test_published_on_loopback_only(self):
        for role in ROLES:
            with self.subTest(role=role):
                ports = podman("port", f"luna-test-{role}").stdout.split()
                self.assertEqual(ports[-1], f"127.0.0.1:{support.PORTS[role]}")

    def test_chat_completes(self):
        r = requests.post(url("chat", "/v1/chat/completions"), headers=self.auth(), timeout=120,
                          json={"messages": [{"role": "user", "content": "Say hello."}],
                                "max_tokens": 16})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertTrue(r.json()["choices"][0]["message"]["content"].strip())

    def test_alias_is_the_repo_name(self):
        r = requests.get(url("chat", "/v1/models"), headers=self.auth(), timeout=5)
        self.assertEqual(r.json()["data"][0]["id"], "Qwen3-0.6B-GGUF")

    def test_embedding_size(self):
        r = requests.post(url("embed", "/v1/embeddings"), headers=self.auth(), timeout=30,
                          json={"input": "The user has a cat named Miso."})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(len(r.json()["data"][0]["embedding"]), 1024)

    def test_downloads_land_in_the_volume(self):
        # LLAMA_CACHE (and HF_HUB_CACHE) point -hf at /models, the volume:
        # a later llama.cpp that ignored both would download elsewhere.
        found = podman("run", "--rm", "-v", f"{support.MODELS}:/models:ro", "--entrypoint", "ls",
                       support.build_llama(), "/models").stdout.split()
        for hf in support.models().values():
            if hf.startswith("ggml-org/SmolVLM"):
                continue                     # only downloaded by LUNA_CONTAINER_TESTS=full
            org, repo = hf.split(":")[0].split("/")
            self.assertIn(f"models--{org}--{repo}", found)

    def test_rerank_puts_the_answer_first(self):
        docs = ["The user likes jazz.", "The user has a cat named Miso.", "Paris is in France."]
        r = requests.post(url("rerank", "/v1/rerank"), headers=self.auth(), timeout=30,
                          json={"query": "what pet does the user have", "documents": docs})
        self.assertEqual(r.status_code, 200, r.text)
        best = max(r.json()["results"], key=lambda x: x["relevance_score"])
        self.assertEqual(best["index"], 1)


if __name__ == "__main__":
    unittest.main()
