"""The installer's model pick with containers (installer/install.py,
pick_model and gpu_reserve): what shares the GPU with her model is kept
free, so an 8 GB card gets the 4B that sees instead of a 9B that won't
fit beside the reranker and the voice. Every other backend: main's pick.

    python -m unittest discover tests        (from the repo folder)
"""
import importlib.util
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.offline import FAKE  # noqa: E402,F401  (first: no real servers)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("luna_installer_cm", os.path.join(ROOT, "installer", "install.py"))
install = importlib.util.module_from_spec(spec)
spec.loader.exec_module(install)

VOICE = {"voice": {"mode": "external", "url": "http://127.0.0.1:8899", "container": "luna-tts"}}


class Reserve(unittest.TestCase):
    def reserve(self, state, nvidia=True, cdi=True):
        with mock.patch.object(install.shutil, "which", return_value="/usr/bin/nvidia-smi" if nvidia else None), \
             mock.patch.object(install.os.path, "exists", side_effect=lambda p: cdi and p.endswith("nvidia.yaml")):
            return install.gpu_reserve(state)

    def test_other_backends_keep_mains_pick(self):
        for backend in ("lmstudio", "llama", "remote", "skip", None):
            self.assertEqual(self.reserve({"backend": backend, **VOICE}), 0)

    def test_containers_keep_room_for_the_reranker(self):
        self.assertEqual(self.reserve({"backend": "container"}), 1.0)

    def test_and_for_a_voice_on_the_gpu(self):
        self.assertEqual(self.reserve({"backend": "container", **VOICE}), 2.0)
        self.assertEqual(self.reserve({"backend": "container", **VOICE}, cdi=False), 1.0)    # voice on the CPU


class Pick(unittest.TestCase):
    def test_eight_gb_with_containers_gets_the_4b_that_sees(self):
        self.assertEqual(install.pick_model(8, 0)[0], "9b-q4")        # main's pick, unchanged
        self.assertEqual(install.pick_model(8, 2.0)[0], "4b")
        self.assertEqual(install.pick_model(8, 1.0)[0], "4b")

    def test_bigger_cards_step_down_one(self):
        self.assertEqual(install.pick_model(16, 0)[0], "9b")
        self.assertEqual(install.pick_model(16, 2.0)[0], "9b")       # 14 + 2 still fits
        self.assertEqual(install.pick_model(24, 2.0)[0], "27b")
        self.assertEqual(install.pick_model(12, 2.0)[0], "9b-q4")

    def test_nothing_fits_is_still_the_smallest(self):
        self.assertEqual(install.pick_model(0, 2.0)[0], "4b-q4")


if __name__ == "__main__":
    unittest.main()
