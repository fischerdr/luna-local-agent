"""faster-whisper decodes speech with PyAV, calling av.open(...,
metadata_errors="ignore") - an argument PyAV 19 removed, so on a fresh
install every spoken turn failed. installer/requirements.txt keeps PyAV
below 19 until faster-whisper stops passing it.

    python -m unittest discover tests        (from the repo folder)
"""
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.offline import FAKE  # noqa: E402,F401  (first: no real servers)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class PyAv(unittest.TestCase):
    def test_requirements_keep_av_below_19(self):
        with open(os.path.join(ROOT, "installer", "requirements.txt"), encoding="utf-8") as f:
            lines = [ln.split("#")[0].strip() for ln in f]
        av = [ln for ln in lines if re.match(r"av\b", ln)]
        self.assertEqual(len(av), 1, "one av line in installer/requirements.txt")
        self.assertRegex(av[0], r"<\s*19\b")

    def test_the_installed_one_takes_metadata_errors(self):
        try:
            import av
            import faster_whisper.audio as fw_audio
        except Exception:
            self.skipTest("faster-whisper isn't installed here")
        import inspect

        if "metadata_errors" in inspect.getsource(fw_audio):
            major = int(av.__version__.split(".")[0])
            self.assertLess(major, 19, f"PyAV {av.__version__} with a faster-whisper that passes metadata_errors")


if __name__ == "__main__":
    unittest.main()
