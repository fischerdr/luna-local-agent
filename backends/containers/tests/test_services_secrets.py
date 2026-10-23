"""The key and token as Podman secrets (services/keys.py): one path inside
every container, replaced only when the value changes - and a change makes
new containers. Podman is pretended; nothing real is created.

    python -m unittest discover tests        (from the repo folder)
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from tests.offline import FAKE  # noqa: E402,F401  (first: no real servers)

from backends.containers import keys  # noqa: E402


class FakePodman:
    """Just enough of podman secret: a store of {name: (value, labels)}."""

    def __init__(self):
        self.store, self.calls = {}, []

    def __call__(self, cmd, input=None, check=False, **kw):
        self.calls.append(cmd)
        assert cmd[:2] == ["podman", "secret"], cmd
        op, name = cmd[2], cmd[-1] if cmd[-1] != "-" else cmd[-2]
        if op == "inspect":
            if name not in self.store:
                return subprocess.CompletedProcess(cmd, 125, "", "no such secret")
            return subprocess.CompletedProcess(cmd, 0, json.dumps(self.store[name][1]), "")
        if op == "create":
            label = cmd[cmd.index("--label") + 1].split("=", 1)
            self.store[name] = (input, {label[0]: label[1]})
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if op == "rm":
            self.store.pop(name, None)
            return subprocess.CompletedProcess(cmd, 0, "", "")
        raise AssertionError(cmd)

    def creates(self):
        return [c for c in self.calls if c[2] == "create"]


class Secrets(unittest.TestCase):
    def setUp(self):
        self.podman = FakePodman()

    def test_value_on_stdin_never_in_argv(self):
        keys.sync("the-key", "hf_the-token", self.podman)
        for cmd in self.podman.calls:
            self.assertNotIn("the-key", " ".join(cmd))
            self.assertNotIn("hf_the-token", " ".join(cmd))
        self.assertEqual(self.podman.store["luna_api_key"][0], "the-key")
        self.assertEqual(self.podman.store["luna_hf_token"][0], "hf_the-token")

    def test_labelled_with_a_fingerprint_not_the_value(self):
        keys.sync("the-key", "", self.podman)
        labels = self.podman.store["luna_api_key"][1]
        self.assertEqual(labels["luna.fingerprint"], keys.fingerprint("the-key"))
        self.assertNotIn("the-key", json.dumps(labels))

    def test_no_token_no_secret(self):
        self.assertEqual(keys.sync("the-key", "", self.podman), ["luna_api_key"])
        self.assertNotIn("luna_hf_token", self.podman.store)

    def test_unchanged_touches_nothing(self):
        keys.sync("the-key", "tok", self.podman)
        before = len(self.podman.creates())
        self.assertEqual(keys.sync("the-key", "tok", self.podman), [])
        self.assertEqual(len(self.podman.creates()), before)

    def test_a_new_key_replaces_the_secret(self):
        keys.sync("the-key", "", self.podman)
        self.assertEqual(keys.sync("new-key", "", self.podman), ["luna_api_key"])
        self.assertIn("--replace", self.podman.creates()[-1])
        self.assertEqual(self.podman.store["luna_api_key"][0], "new-key")

    def test_a_token_gone_removes_its_secret(self):
        keys.sync("the-key", "tok", self.podman)
        self.assertEqual(keys.sync("the-key", "", self.podman), ["luna_hf_token"])
        self.assertNotIn("luna_hf_token", self.podman.store)

    def test_a_new_fingerprint_changes_the_container_hash(self):
        from backends.containers import run, settings

        s = {k: "" for k in settings.KNOWN}
        s.update(settings.DEFAULTS)
        m = settings.Model(hf="org/repo")
        common = dict(image="I", image_id="I", backend="cpu", env_file="/tmp/e", env_text="")
        a = run.run_args("chat", s, m, key_fp=keys.fingerprint("old"), **common)
        b = run.run_args("chat", s, m, key_fp=keys.fingerprint("new"), **common)
        self.assertNotEqual(a[-2], b[-2])


class KeyFile(unittest.TestCase):
    def test_made_like_start_sh_makes_it(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, ".api_key")
            key = keys.make_key(path)
            self.assertEqual(oct(os.stat(path).st_mode & 0o777), "0o600")
            self.assertEqual(keys.read_key(path), key)
            # 32 random bytes in base64 (43 characters and a =), less the
            # / and + taken out: the length varies, so only a floor holds
            self.assertGreaterEqual(len(key), 32)
            self.assertFalse(set(key) & set("/+=\n"))
            self.assertNotEqual(keys.make_key(path), key)      # random, not a pattern

    def test_missing_reads_as_empty(self):
        self.assertEqual(keys.read_key("/nonexistent/.api_key"), "")


if __name__ == "__main__":
    unittest.main()
