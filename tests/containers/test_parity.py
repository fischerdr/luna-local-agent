"""The containers behave as the host servers do, for every feature that
reaches them (plans/container-migration-plan-v2.md, section 10).
"""
import io
import unittest
import wave

import requests

from tests.containers import support
from tests.containers.support import live, url


@live
class Voice(unittest.TestCase):
    """kokoro-reader's contract, as tts/ and Luna use it."""

    @classmethod
    def setUpClass(cls):
        support.start_tts(voice="af_bella")
        try:
            support.wait_ready("tts")
        except Exception:
            support.podman("rm", "-f", "luna-test-tts", check=False)
            raise

    @classmethod
    def tearDownClass(cls):
        support.podman("rm", "-f", "luna-test-tts", check=False)

    def test_health(self):
        r = requests.get(url("tts", "/health"), timeout=5).json()
        self.assertTrue(r["ok"])
        self.assertEqual(r["voice"], "af_bella")      # KOKORO_VOICE from config.json's voice

    def test_voices(self):
        r = requests.get(url("tts", "/voices"), timeout=5).json()
        self.assertIn("af_bella", r["voices"])
        self.assertIn("luna", r["blends"])

    def test_tts_returns_a_wav(self):
        for voice in ("af_bella", "luna"):
            with self.subTest(voice=voice):
                r = requests.post(url("tts", "/tts"), timeout=120,
                                  json={"text": "Hello, this is a test.", "voice": voice, "speed": 1.0})
                self.assertEqual(r.status_code, 200, r.text[:200])
                self.assertEqual(r.headers["Content-Type"], "audio/wav")
                w = wave.open(io.BytesIO(r.content))
                self.assertEqual((w.getnchannels(), w.getframerate()), (1, 24000))
                self.assertGreater(w.getnframes(), 24000 // 2)

    def test_stops_promptly(self):
        # --init: the server as PID 1 would sit out SIGTERM's 10 seconds.
        out = support.podman("inspect", "-f", "{{.HostConfig.Init}}", "luna-test-tts").stdout.strip()
        self.assertEqual(out, "true")


TOOL = {"type": "function", "function": {"name": "get_datetime", "description": "The current date and time.",
                                         "parameters": {"type": "object", "properties": {}}}}


@live
class Llama(unittest.TestCase):
    """llama-server in its container, asked what Luna asks it."""

    @classmethod
    def setUpClass(cls):
        m = support.models()
        cls.key = support.make_key()
        try:
            env = {"CTX": "4096", "PARALLEL": "2", "NGL": "0", "ALIAS": "luna-test-model"}
            support.start("chat", m["TEST_CHAT_HF"], env)
            support.wait_ready("chat")
        except Exception:
            support.cleanup("chat")
            raise

    @classmethod
    def tearDownClass(cls):
        support.cleanup("chat")

    def post(self, body, **kw):
        return requests.post(url("chat", "/v1/chat/completions"), timeout=180,
                             headers={"Authorization": f"Bearer {self.key}", **kw.pop("headers", {})},
                             json=body, **kw)

    def chat(self, text, **more):
        return self.post({"messages": [{"role": "user", "content": text}], "max_tokens": 64, **more})

    def test_token_confidence_streamed(self):
        r = self.post({"messages": [{"role": "user", "content": "Say hello."}], "max_tokens": 8,
                       "stream": True, "n_probs": 3}, stream=True)
        self.assertEqual(r.status_code, 200)
        import json

        probs = []
        for line in r.iter_lines():
            if line.startswith(b"data: ") and line != b"data: [DONE]":
                choice = (json.loads(line[6:]).get("choices") or [{}])[0]
                if choice.get("logprobs"):
                    probs += choice["logprobs"].get("content") or []
        self.assertTrue(probs, "no per-token probabilities in the stream")
        self.assertIn("top_logprobs", probs[0])

    def test_thinking_budget(self):
        off = self.chat("What is 2+2?", reasoning_budget_tokens=0).json()["choices"][0]["message"]
        self.assertFalse((off.get("reasoning_content") or "").strip())
        on = self.chat("What is 17*3? Think it through.", max_tokens=200).json()["choices"][0]["message"]
        self.assertTrue((on.get("reasoning_content") or "").strip(), "no reasoning_content with no budget")

    def test_tool_calls(self):
        r = self.chat("What time is it? Use your tool.", tools=[TOOL], max_tokens=300,
                      reasoning_budget_tokens=0).json()
        calls = r["choices"][0]["message"].get("tool_calls") or []
        self.assertTrue(calls, r)
        self.assertEqual(calls[0]["function"]["name"], "get_datetime")

    def test_the_model_is_named_by_alias(self):
        r = requests.get(url("chat", "/v1/models"), headers={"Authorization": f"Bearer {self.key}"}, timeout=5)
        self.assertEqual(r.json()["data"][0]["id"], "luna-test-model")

    def test_key_and_cors(self):
        self.assertEqual(requests.post(url("chat", "/v1/chat/completions"), json={}, timeout=5).status_code, 401)
        here = requests.options(url("chat", "/v1/chat/completions"), timeout=5,
                                headers={"Origin": "http://localhost", "Access-Control-Request-Method": "POST"})
        away = requests.options(url("chat", "/v1/chat/completions"), timeout=5,
                                headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "POST"})
        self.assertIn("localhost", here.headers.get("Access-Control-Allow-Origin", ""))
        self.assertNotIn("evil.example", away.headers.get("Access-Control-Allow-Origin", ""))

    def test_props_show_ctx_and_parallel(self):
        p = requests.get(url("chat", "/props"), headers={"Authorization": f"Bearer {self.key}"}, timeout=5).json()
        self.assertEqual(p["total_slots"], 2)
        ctx = p["default_generation_settings"].get("n_ctx")
        self.assertIn(ctx, (4096, 2048))          # KV_UNIFIED shares CTX, else a share per slot
        self.assertFalse(p["modalities"]["vision"])

    def test_slots_save_and_restore(self):
        self.chat("Remember the word banana.")
        auth = {"Authorization": f"Bearer {self.key}"}
        saved = requests.post(url("chat", "/slots/0?action=save"), json={"filename": "parity.bin"},
                              headers=auth, timeout=30)
        self.assertEqual(saved.status_code, 200, saved.text)
        restored = requests.post(url("chat", "/slots/0?action=restore"), json={"filename": "parity.bin"},
                                 headers=auth, timeout=30)
        self.assertEqual(restored.status_code, 200, restored.text)
        self.assertIn("parity.bin", support.podman("exec", "luna-test-chat", "ls", "/data/slots").stdout)


@support.full
class Vision(unittest.TestCase):
    """LUNA_CONTAINER_TESTS=full: a small vision model and its mmproj."""

    def props(self, env):
        support.make_key()
        try:
            support.start("chat", support.models()["TEST_VISION_HF"], dict({"CTX": "2048", "NGL": "0"}, **env))
            support.wait_ready("chat")
            key = support.podman("exec", "luna-test-chat", "cat", "/run/secrets/api_key").stdout
            return requests.get(url("chat", "/props"), headers={"Authorization": f"Bearer {key}"}, timeout=5).json()
        finally:
            support.cleanup("chat")

    def test_vision_with_its_mmproj(self):
        self.assertTrue(self.props({})["modalities"]["vision"])

    def test_mmproj_off(self):
        self.assertFalse(self.props({"MMPROJ": "off"})["modalities"]["vision"])


if __name__ == "__main__":
    unittest.main()
