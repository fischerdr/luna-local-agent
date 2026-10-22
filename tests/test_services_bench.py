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
    """Chat and her voice get the GPU first; the reranker, then the embedder,
    only if there's room after them - else the CPU, never off."""

    def test_room_for_everything(self):
        place, _ = bench.recommend(24576, 9000, 1000, 0.5, voice_need=1000, embed_need=1000)
        self.assertEqual(place, {"EMBED_NGL": 99, "RERANK_NGL": 99, "RERANK": "auto"})

    def test_voice_before_the_side_models(self):
        # 8 GB, 4B chat (~5.1 GB): room for the voice, not for the reranker too
        place, why = bench.recommend(8192, 5600, 1000, 0.8, voice_need=1000, embed_need=1000)
        self.assertNotIn("TTS_DEVICE", place)
        self.assertEqual((place["RERANK_NGL"], place["EMBED_NGL"]), (0, 0))
        self.assertIn("voice on the GPU", why[0])

    def test_reranker_before_the_embedder(self):
        place, _ = bench.recommend(8192, 4000, 1000, 0.8, voice_need=1000, embed_need=1500)
        self.assertEqual((place["RERANK_NGL"], place["EMBED_NGL"]), (99, 0))

    def test_ten_percent_kept_free(self):
        # 6500 + 1000 = 7500 > 90% of 8192 (7372.8)
        self.assertEqual(bench.recommend(8192, 6500, 1000, 0.8)[0]["RERANK_NGL"], 0)
        self.assertEqual(bench.recommend(8192, 6300, 1000, 0.8)[0]["RERANK_NGL"], 99)

    def test_never_off_a_slow_cpu_only_says_so(self):
        place, why = bench.recommend(8192, 7900, 1000, 2.4)
        self.assertEqual((place["RERANK_NGL"], place["RERANK"]), (0, "auto"))
        self.assertIn("embedding order", " ".join(why))

    def test_no_room_for_the_voice(self):
        place, why = bench.recommend(8192, 6900, 1000, 0.5, voice_need=1000)
        self.assertEqual((place["TTS_DEVICE"], place["RERANK_NGL"]), ("cpu", 0))

    def test_chat_alone_too_big_says_so(self):
        place, why = bench.recommend(8192, 7600, 1000, 0.5, voice_need=1000, embed_need=1000)
        self.assertEqual((place["TTS_DEVICE"], place["RERANK_NGL"], place["EMBED_NGL"]), ("cpu", 0, 0))
        self.assertIn("chat alone", why[0])
        self.assertIn("--model 4b", why[0])

    def test_no_gpu(self):
        place, _ = bench.recommend(None, None, 1000, 5.0, embed_need=1000)
        self.assertEqual(place, {"EMBED_NGL": 0, "RERANK_NGL": 0, "RERANK": "auto"})

    def test_a_cpu_voice_isnt_counted(self):
        self.assertNotIn("TTS_DEVICE", bench.recommend(8192, 6900, 1000, 0.5, voice_need=0)[0])


class ModelSize(unittest.TestCase):
    def test_not_downloaded_yet_counts_its_download(self):
        from services import settings

        m = settings.Model(hf="bartowski/Qwen_Qwen3.5-9B-GGUF:Q4_K_M")
        with mock.patch.object(bench.lifecycle, "podman") as podman, \
             mock.patch("services.setup.hf_size", return_value=6_200_000_000):
            self.assertEqual(bench.model_mib(m, {}), 6_200_000_000 // bench.MIB)
        podman.assert_not_called()        # never measured in the volume (the bug: du of nothing)


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
