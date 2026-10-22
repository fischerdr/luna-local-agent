"""Where the side models go (services/bench.py): GPU, CPU or off, from a
pretended GPU and pretended timings.

    python -m unittest discover tests        (from the repo folder)
"""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.offline import FAKE  # noqa: E402,F401  (first: no real servers)

from services import bench  # noqa: E402


class Recommend(unittest.TestCase):
    def test_room_on_the_gpu(self):
        place, why = bench.recommend(8192, 5800, 1000, 3.0)
        self.assertEqual(place, {"EMBED_NGL": 0, "RERANK_NGL": 99, "RERANK": "auto"})
        self.assertIn("on the GPU", why[-1])

    def test_ten_percent_kept_free(self):
        # 6500 + 1000 = 7500 > 90% of 8192 (7372.8)
        place, _ = bench.recommend(8192, 6500, 1000, 0.8)
        self.assertEqual((place["RERANK_NGL"], place["RERANK"]), (0, "auto"))

    def test_cpu_when_quick_enough(self):
        place, why = bench.recommend(8192, 7900, 1000, 1.2)
        self.assertEqual(place, {"EMBED_NGL": 0, "RERANK_NGL": 0, "RERANK": "auto"})
        self.assertIn("1.20 s", why[-1])

    def test_off_when_too_slow(self):
        place, why = bench.recommend(8192, 7900, 1000, 2.4)
        self.assertEqual((place["RERANK_NGL"], place["RERANK"]), (0, "off"))
        self.assertIn("embedding order", why[-1])

    def test_off_when_it_couldnt_be_timed(self):
        self.assertEqual(bench.recommend(8192, 7900, 1000, None)[0]["RERANK"], "off")

    def test_no_gpu(self):
        self.assertEqual(bench.recommend(None, None, 1000, 0.5)[0]["RERANK_NGL"], 0)
        self.assertEqual(bench.recommend(None, None, 1000, 5.0)[0]["RERANK"], "off")

    def test_embedder_always_on_the_cpu(self):
        for args in ((8192, 1000, 500, 0.1), (24576, 2000, 500, 0.1), (None, None, 500, 9.0)):
            self.assertEqual(bench.recommend(*args)[0]["EMBED_NGL"], 0)


class Voice(unittest.TestCase):
    def test_everything_fits(self):
        place, _ = bench.recommend(24576, 9000, 1000, 0.5, voice_need=1000)
        self.assertNotIn("TTS_DEVICE", place)
        self.assertEqual(place["RERANK_NGL"], 99)

    def test_tight_card_moves_the_voice_first(self):
        # the zero test's 8 GB card: a 9B Q4 chat (~6.8 GB with its context)
        place, why = bench.recommend(8192, 6000, 1000, 0.5, voice_need=1000)
        self.assertEqual(place["TTS_DEVICE"], "cpu")
        self.assertEqual(place["RERANK_NGL"], 99)            # 6000 + 1000 still fits once the voice moves

    def test_tighter_moves_both(self):
        place, _ = bench.recommend(8192, 6900, 1000, 0.5, voice_need=1000)
        self.assertEqual((place["TTS_DEVICE"], place["RERANK_NGL"], place["RERANK"]), ("cpu", 0, "auto"))

    def test_chat_alone_too_big_says_so(self):
        place, why = bench.recommend(8192, 7600, 1000, 0.5, voice_need=1000)
        self.assertEqual((place["TTS_DEVICE"], place["RERANK_NGL"]), ("cpu", 0))
        self.assertIn("chat alone", why[0])
        self.assertIn("--model 4b", why[0])

    def test_a_cpu_voice_isnt_counted(self):
        self.assertNotIn("TTS_DEVICE", bench.recommend(8192, 6900, 1000, 0.5, voice_need=0)[0])


class ModelSize(unittest.TestCase):
    def test_not_downloaded_yet_counts_its_download(self):
        from services import settings

        m = settings.Model(hf="bartowski/Qwen_Qwen3.5-9B-GGUF:Q4_K_M")
        with mock.patch.object(bench.lifecycle, "podman", return_value=mock.Mock(stdout="0\n")), \
             mock.patch.object(bench.images, "find", return_value="IMG"), \
             mock.patch("services.setup.hf_size", return_value=6_200_000_000):
            self.assertEqual(bench.model_mib(m, {}), 6_200_000_000 // bench.MIB)


class Vram(unittest.TestCase):
    def test_nvidia_smi(self):
        run = mock.Mock(return_value=mock.Mock(stdout="8192, 7268\n"))
        with mock.patch.object(bench.shutil, "which", side_effect=lambda n: n == "nvidia-smi" and "/usr/bin/x"):
            self.assertEqual(bench.vram(run), (8192, 7268))

    def test_none(self):
        with mock.patch.object(bench.shutil, "which", return_value=None):
            self.assertIsNone(bench.vram())


if __name__ == "__main__":
    unittest.main()
