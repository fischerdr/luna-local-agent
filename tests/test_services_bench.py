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
        (embed, rerank, switch), why = bench.recommend(8192, 5800, 1000, 3.0)
        self.assertEqual((embed, rerank, switch), (0, 99, "auto"))
        self.assertIn("on the GPU", why)

    def test_ten_percent_kept_free(self):
        # 6500 + 1000 = 7500 > 90% of 8192 (7372.8)
        (_, rerank, switch), _ = bench.recommend(8192, 6500, 1000, 0.8)
        self.assertEqual((rerank, switch), (0, "auto"))

    def test_cpu_when_quick_enough(self):
        (embed, rerank, switch), why = bench.recommend(8192, 7900, 1000, 1.2)
        self.assertEqual((embed, rerank, switch), (0, 0, "auto"))
        self.assertIn("1.20 s", why)

    def test_off_when_too_slow(self):
        (_, rerank, switch), why = bench.recommend(8192, 7900, 1000, 2.4)
        self.assertEqual((rerank, switch), (0, "off"))
        self.assertIn("embedding order", why)

    def test_off_when_it_couldnt_be_timed(self):
        self.assertEqual(bench.recommend(8192, 7900, 1000, None)[0][2], "off")

    def test_no_gpu(self):
        self.assertEqual(bench.recommend(None, None, 1000, 0.5)[0], (0, 0, "auto"))
        self.assertEqual(bench.recommend(None, None, 1000, 5.0)[0], (0, 0, "off"))

    def test_embedder_always_on_the_cpu(self):
        for args in ((8192, 1000, 500, 0.1), (24576, 2000, 500, 0.1), (None, None, 500, 9.0)):
            self.assertEqual(bench.recommend(*args)[0][0], 0)


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
