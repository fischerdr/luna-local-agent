"""Finding your Hugging Face token (services/keys.py): Hugging Face's own
order, read only - never copied or written anywhere. A temporary HOME.

    python -m unittest discover tests        (from the repo folder)
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from tests.offline import FAKE  # noqa: E402,F401  (first: no real servers)

from backends.containers import keys  # noqa: E402


class Lookup(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def put(self, rel, text):
        path = os.path.join(self.home, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(text)
        return path

    def files(self):
        out = []
        for d, _, fs in os.walk(self.home):
            out += [os.path.join(d, f) for f in fs]
        return sorted(out)

    def test_env_first(self):
        self.put(".cache/huggingface/token", "from-file\n")
        self.assertEqual(keys.token_source({"HF_TOKEN": "from-env"}, self.home), ("from-env", "HF_TOKEN"))

    def test_hf_home_before_the_default(self):
        hf_home = os.path.join(self.home, "hf")
        path = self.put("hf/token", "from-hf-home\n")
        self.put(".cache/huggingface/token", "from-default\n")
        self.assertEqual(keys.token_source({"HF_HOME": hf_home}, self.home), ("from-hf-home", path))

    def test_the_default_place(self):
        path = self.put(".cache/huggingface/token", "from-default\n")
        self.assertEqual(keys.token_source({}, self.home), ("from-default", path))

    def test_none_is_fine(self):
        self.assertEqual(keys.token_source({}, self.home), ("", ""))
        self.assertEqual(keys.token_source({"HF_TOKEN": "  "}, self.home), ("", ""))

    def test_never_written(self):
        self.put(".cache/huggingface/token", "tok\n")
        before = self.files()
        keys.token_source({}, self.home)
        self.assertEqual(self.files(), before)


if __name__ == "__main__":
    unittest.main()
