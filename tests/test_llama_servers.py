"""Integration test for the llama.cpp server pipeline.

Verifies that all three models load and respond:
  - Chat model  on :8080
  - Embedding   on :8081
  - Reranker    on :8082

And that the embedding → reranker retrieval pipeline works end-to-end.

Run from the project root:
    pytest tests/test_llama_servers.py -v
"""

import os
import sys
import time
import subprocess
import unittest

import requests

CHAT_URL = "http://127.0.0.1:8080"
EMBED_URL = "http://127.0.0.1:8081"
RERANK_URL = "http://127.0.0.1:8082"
HEADERS = {"Content-Type": "application/json"}
CHAT_MODEL = os.environ.get("CHAT_MODEL", "qwen3.5_4b_uncensored.Q8_0")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def wait_for(url, label, timeout=120, interval=2):
    """Block until *url* returns 200 on /health (or /v1/models for embed/rerank)."""
    path = "/health" if "8080" in url else "/v1/models"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            r = requests.get(url + path, headers=HEADERS, timeout=5)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(interval)
    raise TimeoutError(f"{label} did not become ready within {timeout}s")


def chat_completion(prompt, max_tokens=64):
    """Send a chat completion request. Returns the parsed JSON."""
    r = requests.post(
        f"{CHAT_URL}/v1/chat/completions",
        headers=HEADERS,
        json={
            "model": CHAT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": False,
        },
        timeout=120,
    )
    r.raise_for_status()
    return r.json()


def embed_text(texts):
    """Embed one or more texts. Returns list of embedding dicts."""
    r = requests.post(
        f"{EMBED_URL}/v1/embeddings",
        headers=HEADERS,
        json={"input": texts, "model": "embed"},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["data"]


def rerank_documents(query, documents):
    """Rerank documents against a query. Returns parsed JSON."""
    r = requests.post(
        f"{RERANK_URL}/v1/rerank",
        headers=HEADERS,
        json={
            "model": "rerank",
            "query": query,
            "documents": documents,
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestChatModel(unittest.TestCase):
    """The chat model loads and answers on port 8080."""

    @classmethod
    def setUpClass(cls):
        wait_for(CHAT_URL, "chat model")

    def test_health(self):
        r = requests.get(f"{CHAT_URL}/health", timeout=5)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "ok")

    def test_v1_models(self):
        r = requests.get(f"{CHAT_URL}/v1/models", headers=HEADERS, timeout=5)
        self.assertEqual(r.status_code, 200)
        data = r.json().get("data", [])
        self.assertGreater(len(data), 0, "no models listed")

    def test_chat_simple(self):
        """Chat model responds to a simple prompt."""
        resp = chat_completion("Reply with just the word 'hello'.", max_tokens=10)
        msg = resp.get("choices", [{}])[0].get("message", {})
        content = msg.get("content", "")
        reasoning = msg.get("reasoning_content", "")
        self.assertTrue(content or reasoning)

    def test_chat_math(self):
        """Chat model handles simple arithmetic."""
        resp = chat_completion("What is 1 plus 1? Answer with just the number.",
                               max_tokens=64)
        msg = resp.get("choices", [{}])[0].get("message", {})
        content = msg.get("content", "")
        reasoning = msg.get("reasoning_content", "")
        combined = (content + " " + reasoning).lower()
        self.assertTrue("2" in combined or "two" in combined,
                        f"expected '2' in response, got: {combined[:300]}")


class TestEmbeddingModel(unittest.TestCase):
    """The embedding model loads and generates vectors on port 8081."""

    @classmethod
    def setUpClass(cls):
        # Start embedding on-demand (no-op if already running)
        from llama import llama_side
        llama_side.ensure_embed()
        wait_for(EMBED_URL, "embedding model", timeout=90)

    def test_models_list(self):
        r = requests.get(f"{EMBED_URL}/v1/models", headers=HEADERS, timeout=5)
        self.assertEqual(r.status_code, 200)
        data = r.json().get("data", [])
        self.assertGreater(len(data), 0)

    def test_embed_single(self):
        embeddings = embed_text(["hello world"])
        self.assertEqual(len(embeddings), 1)
        emb = embeddings[0]["embedding"]
        self.assertGreater(len(emb), 0, "empty embedding vector")
        # vectors are typically 1–4096 dims for small models

    def test_embed_multiple(self):
        embeddings = embed_text(["hello world", "goodbye world", "foo bar"])
        self.assertEqual(len(embeddings), 3)

    def test_embed_similarity(self):
        """Similar texts should have more similar vectors."""
        emb = embed_text([
            "the quick brown fox",
            "a fast brown fox leaping",
            "computers and programming",
        ])
        # Embedding 0 and 1 are similar; 2 is different
        import math
        def dot(a, b):
            return sum(x * y for x, y in zip(a, b))
        sim_close = dot(emb[0]["embedding"], emb[1]["embedding"])
        sim_far = dot(emb[0]["embedding"], emb[2]["embedding"])
        self.assertGreater(sim_close, sim_far,
                           f"similar texts should score higher: close={sim_close:.3f} far={sim_far:.3f}")


class TestRerankerModel(unittest.TestCase):
    """The reranker model loads and re-orders documents on port 8082."""

    @classmethod
    def setUpClass(cls):
        # Start reranker on-demand (no-op if already running)
        from llama import llama_side
        llama_side.ensure_rerank()
        wait_for(RERANK_URL, "reranker model", timeout=90)

    def test_models_list(self):
        r = requests.get(f"{RERANK_URL}/v1/models", headers=HEADERS, timeout=5)
        self.assertEqual(r.status_code, 200)
        data = r.json().get("data", [])
        self.assertGreater(len(data), 0)

    def test_rerank(self):
        results = rerank_documents(
            "how to change the oil",
            [
                "to change the oil, first drain the old oil from the engine",
                "the best coffee shops in Portland",
                "python programming is fun and useful",
            ],
        )
        self.assertIn("results", results)
        self.assertEqual(len(results["results"]), 3)
        # The oil document should rank first
        best_index = results["results"][0]["index"]
        self.assertEqual(best_index, 0,
                         f"expected 'oil' doc (index 0) to rank first, got index {best_index}")

    def test_rerank_scores(self):
        """Re-ranked results should have relevance scores."""
        results = rerank_documents(
            "machine learning algorithms",
            [
                "neural networks learn patterns from data",
                "buy cheap shoes online",
                "the weather is nice today",
            ],
        )
        for r in results["results"]:
            self.assertIn("relevance_score", r)
            score = r["relevance_score"]
            self.assertIsInstance(score, float)


class TestRetrievalPipeline(unittest.TestCase):
    """End-to-end: embed → rank → rerank → select (the full Luna knowledge flow)."""

    @classmethod
    def setUpClass(cls):
        # Start both on-demand
        from llama import llama_side
        llama_side.ensure_embed()
        llama_side.ensure_rerank()
        wait_for(EMBED_URL, "embedding model", timeout=90)
        wait_for(RERANK_URL, "reranker model", timeout=90)

    def test_full_pipeline(self):
        """Query: 'best laptop for programming' should rank a tech doc highest."""
        docs = [
            "the best laptops for programming have at least 16 GB RAM and a good keyboard",
            "how to make chocolate chip cookies",
            "professional soccer teams in europe",
            "python is a great language for web development and data science",
        ]

        # Embed all docs
        embeddings = embed_text(docs)
        self.assertEqual(len(embeddings), 4)

        # Rerank
        results = rerank_documents("best laptop for programming", docs)
        ordered = [docs[r["index"]] for r in results["results"]]

        # The laptop doc or the python doc should be in the top 2
        top2 = ordered[:2]
        has_tech = any("laptop" in d or "programming" in d or "python" in d for d in top2)
        self.assertTrue(has_tech,
                        f"top-2 should include a tech doc: {top2}")

    def test_reranker_over_embedding(self):
        """Reranker should distinguish relevant docs better than embedding alone.

        The embedding model might confuse 'apple' in different contexts.
        The reranker reads the query + doc together, so it's sharper.
        """
        docs = [
            "apple released a new iphone today",
            "apple pie recipe with cinnamon",
            "apple stock price fell 5 percent",
            "macintosh apple computers were first sold in 1984",
        ]
        results = rerank_documents("best apple for baking", docs)
        ordered = [docs[r["index"]] for r in results["results"]]
        # The pie recipe should be first
        self.assertIn("pie", ordered[0].lower(),
                      f"expected pie recipe first, got: {ordered[0]}")


class TestContainerRunning(unittest.TestCase):
    """Verify the podman container is running."""

    def test_container_running(self):
        r = subprocess.run(
            ["podman", "ps", "--filter", "name=luna-llama",
             "--format", "{{.Status}}"],
            capture_output=True, text=True,
        )
        self.assertIn("Up", r.stdout, "luna-llama container is not running")


if __name__ == "__main__":
    unittest.main()
