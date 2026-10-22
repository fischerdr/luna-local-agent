"""--remote: her servers on another machine (installer/install.py). The
address, the URLs written to config.json, the key, and what the probe says.
A temporary repo folder; the network is pretended.

    python -m unittest discover tests        (from the repo folder)
"""
import contextlib
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
import urllib.error
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.offline import FAKE  # noqa: E402,F401  (first: no real servers)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("luna_installer_remote", os.path.join(ROOT, "installer", "install.py"))
install = importlib.util.module_from_spec(spec)
spec.loader.exec_module(install)


class Address(unittest.TestCase):
    def test_good(self):
        for addr in ("192.168.1.50", "10.0.0.2", "fd00::5", "gpu-box", "gpu-box.lan"):
            with self.subTest(addr=addr):
                self.assertIsNone(install.check_address(addr))

    def test_bad(self):
        for addr in ("127.0.0.1", "127.1.2.3", "::1", "localhost", "0.0.0.0", "a b", "-x.lan", "x..lan"):
            with self.subTest(addr=addr):
                self.assertIsNotNone(install.check_address(addr))

    def test_url_host(self):
        self.assertEqual(install.url_host("192.168.1.50"), "192.168.1.50")
        self.assertEqual(install.url_host("fd00::5"), "[fd00::5]")
        self.assertEqual(install.url_host("gpu-box.lan"), "gpu-box.lan")


class Response:
    def __init__(self, status):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def answering(codes):
    """A urlopen that answers each port with its code (None = no answer)."""
    def get(req, timeout):
        port = int(req.full_url.split(":")[-1].split("/")[0])
        code = codes.get(port)
        if code is None:
            raise urllib.error.URLError("refused")
        if code >= 400:
            raise urllib.error.HTTPError(req.full_url, code, "no", {}, io.BytesIO())
        return Response(code)
    return get


class Probe(unittest.TestCase):
    def test_all_up(self):
        lines = install.remote_probe("10.0.0.2", "k", get=answering({8080: 200, 8081: 200, 8082: 200, 8899: 200}))
        self.assertTrue(all(good for _n, good, _l in lines))
        self.assertEqual([n for n, _g, _l in lines], ["chat", "embeddings", "reranker", "voice"])

    def test_wrong_key(self):
        lines = install.remote_probe("10.0.0.2", "k", get=answering({8080: 401, 8081: 401, 8082: 401, 8899: 200}))
        self.assertIn("key doesn't match the server's", lines[0][2])
        self.assertTrue(lines[3][1])

    def test_no_answer_says_what_to_check(self):
        with mock.patch.object(install, "client_ip", return_value="10.0.0.9"):
            lines = install.remote_probe("10.0.0.2", "k", get=answering({}))
        self.assertFalse(lines[0][1])
        for words in ("server is up", "LUNA_BIND", "firewall", "10.0.0.9"):
            self.assertIn(words, lines[0][2])

    def test_key_goes_as_a_bearer_header(self):
        seen = []

        def get(req, timeout):
            seen.append(req.headers.get("Authorization"))
            return Response(200)
        install.remote_probe("10.0.0.2", "the-key", get=get)
        self.assertEqual(seen[0], "Bearer the-key")


class Setup(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        os.makedirs(os.path.join(self.tmp.name, "llama"))
        shutil.copy(os.path.join(ROOT, "config.example.json"), os.path.join(self.tmp.name, "config.example.json"))
        p = mock.patch.object(install, "ROOT", self.tmp.name)
        p.start()
        self.addCleanup(p.stop)
        install.REPORT.clear()

    def config(self):
        with open(os.path.join(self.tmp.name, "config.json")) as f:
            return json.load(f)

    def key_path(self):
        return os.path.join(self.tmp.name, "llama", ".api_key")

    def run_remote(self, host, answers=(), yes=False, codes=None):
        install.ARGS = install.parser().parse_args(["--remote", host] + (["--yes"] if yes else []))
        codes = codes if codes is not None else {8080: 200, 8081: 200, 8082: 200, 8899: 200}
        with mock.patch("builtins.input", side_effect=list(answers)), \
             mock.patch.object(install.urllib.request, "urlopen", side_effect=answering(codes)), \
             contextlib.redirect_stdout(io.StringIO()) as out:
            install.remote({})
        return out.getvalue()

    def test_urls_and_autostart(self):
        self.run_remote("192.168.1.50", answers=["abc"])
        llm = self.config()["llm"]
        self.assertEqual(llm["backend"], "llama")
        self.assertEqual(llm["llama_url"], "http://192.168.1.50:8080/v1/chat/completions")
        self.assertEqual(llm["embed_url"], "http://192.168.1.50:8081/v1/embeddings")
        self.assertEqual(llm["rerank_url"], "http://192.168.1.50:8082/v1/rerank")
        self.assertIs(llm["autostart"], False)

    def test_ipv6_urls(self):
        self.run_remote("fd00::5", answers=["abc"])
        self.assertEqual(self.config()["llm"]["llama_url"], "http://[fd00::5]:8080/v1/chat/completions")

    def test_key_is_private(self):
        self.run_remote("192.168.1.50", answers=["the-servers-key"])
        with open(self.key_path()) as f:
            self.assertEqual(f.read(), "the-servers-key")
        self.assertEqual(oct(os.stat(self.key_path()).st_mode & 0o777), "0o600")

    def test_key_from_a_file(self):
        src = os.path.join(self.tmp.name, "api_key")
        with open(src, "w") as f:
            f.write("from-file\n")
        self.run_remote("192.168.1.50", answers=[src])
        with open(self.key_path()) as f:
            self.assertEqual(f.read(), "from-file")

    def test_asks_before_replacing_a_key(self):
        install.write_key("old-key")
        self.run_remote("192.168.1.50", answers=["new-key", "n"])
        with open(self.key_path()) as f:
            self.assertEqual(f.read(), "old-key")
        self.run_remote("192.168.1.50", answers=["new-key", "y"])
        with open(self.key_path()) as f:
            self.assertEqual(f.read(), "new-key")

    def test_yes_never_invents_a_key(self):
        out = self.run_remote("192.168.1.50", yes=True)
        self.assertFalse(os.path.exists(self.key_path()))
        self.assertIn("deploy/api_key", out)

    def test_report_follows_the_probe(self):
        self.run_remote("192.168.1.50", answers=["k"], codes={})
        self.assertFalse(install.REPORT[-1][1])


if __name__ == "__main__":
    unittest.main()
