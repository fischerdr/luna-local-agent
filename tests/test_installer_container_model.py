"""The installer's model pick with containers (installer/install.py,
pick_model and gpu_reserve): chat and her voice come first on the GPU, so
the voice's room is kept free - an 8 GB card gets the 4B that sees, not
a 9B that won't fit beside her voice. The reranker and embedder take what
is left, or the CPU. Every other backend, remote included: main's pick.

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

    def test_containers_keep_room_for_a_voice_on_the_gpu(self):
        self.assertEqual(self.reserve({"backend": "container", **VOICE}), 1.0)

    def test_nothing_kept_for_the_side_models(self):
        self.assertEqual(self.reserve({"backend": "container"}), 0)                   # no voice container
        self.assertEqual(self.reserve({"backend": "container", **VOICE}, cdi=False), 0)    # voice on the CPU


class Pick(unittest.TestCase):
    def test_eight_gb_with_containers_gets_the_4b_that_sees(self):
        self.assertEqual(install.pick_model(8, 0)[0], "9b-q4")        # main's pick, unchanged
        self.assertEqual(install.pick_model(8, 1.0)[0], "4b")

    def test_bigger_cards_step_down_one(self):
        self.assertEqual(install.pick_model(16, 0)[0], "9b")
        self.assertEqual(install.pick_model(16, 1.0)[0], "9b")       # 14 + 1 still fits
        self.assertEqual(install.pick_model(24, 1.0)[0], "27b")
        self.assertEqual(install.pick_model(11, 1.0)[0], "9b-q4")

    def test_nothing_fits_is_still_the_smallest(self):
        self.assertEqual(install.pick_model(0, 1.0)[0], "4b-q4")


class ModelHf(unittest.TestCase):
    """The pick becomes MODEL_HF for the containers - the uncensored files
    (main's default) name their quant after a dot, Qwen's after a dash."""

    def model_hf(self, key, uncensored):
        import contextlib
        import io
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "services"))
            os.makedirs(os.path.join(tmp, "llama"))
            row = install.variant(install._model_by_key(key), uncensored)
            k, label, repo, file, size, mmproj, _least, ctx, _lms = row
            with mock.patch.object(install, "ROOT", tmp), contextlib.redirect_stdout(io.StringIO()):
                install.container_model({}, k, label, repo, file, mmproj, ctx)
            with open(os.path.join(tmp, "services", "services.local.env")) as f:
                return [ln for ln in f.read().splitlines() if ln.startswith("MODEL_HF=")][0]

    def test_every_model_both_flavours(self):
        for m in install.MODELS:
            for uncensored in (True, False):
                with self.subTest(model=m[0], uncensored=uncensored):
                    row = install.variant(m, uncensored)
                    quant = row[3].rsplit(".", 1)[0].replace("-", ".").rsplit(".", 1)[-1]
                    self.assertEqual(self.model_hf(m[0], uncensored), f'MODEL_HF="{row[2]}:{quant}"')

    def test_the_8gb_pick(self):
        self.assertEqual(self.model_hf("4b", True),
                         'MODEL_HF="mradermacher/Huihui-Qwen3.5-4B-abliterated-GGUF:Q6_K"')


if __name__ == "__main__":
    unittest.main()
