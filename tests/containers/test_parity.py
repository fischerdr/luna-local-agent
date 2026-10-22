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


if __name__ == "__main__":
    unittest.main()
