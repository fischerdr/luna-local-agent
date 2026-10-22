"""Bringing a role up (services/lifecycle.py): a container is kept when its
config hash matches and made anew when not; one that isn't ours is never
replaced; a busy port is named and left alone; readiness is polled, never
slept for. Podman, ss and HTTP are all pretended.

    python -m unittest discover tests        (from the repo folder)
"""
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.offline import FAKE  # noqa: E402,F401  (first: no real servers)

import requests  # noqa: E402

from services import lifecycle, settings  # noqa: E402


def base_settings(**more):
    s = {k: "" for k in settings.KNOWN}
    s.update(settings.DEFAULTS)
    s.update({"EMBED_HF": "org/embed:Q8_0", "RERANK_HF": "org/rerank:Q8_0", "MODEL_HF": "org/chat:Q4",
              "NGL": "99", "EMBED_NGL": "0", "RERANK_NGL": "99", "THREADS": "9", "EMBED_POOLING": "last"})
    s.update(more)
    return s


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patches = [mock.patch.object(lifecycle, "env_dir", return_value=self.tmp.name),
                   mock.patch.object(lifecycle, "port_holder", return_value=None)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.ran = []
        run = mock.patch.object(lifecycle.subprocess, "run", side_effect=self.fake_run)
        run.start()
        self.addCleanup(run.stop)
        self.ctx = lifecycle.Context(backend="cuda13", key_fp="kfp", image_ids={"cuda13": "IMGGPU", "cpu": "IMGCPU"})

    def fake_run(self, cmd, **kw):
        self.ran.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    def runs(self):
        return [c for c in self.ran if c[:1] == ["podman"] and "run" in c[:4]]


class KeepOrReplace(Base):
    def wanted_hash(self, role, s):
        argv, _, _ = lifecycle.plan(role, s, self.ctx)
        return argv[-2].split("=", 1)[1]

    def test_first_up_starts_it(self):
        r = lifecycle.ensure("embed", base_settings(), self.ctx, found={})
        self.assertEqual((r.state, r.note), ("started", "org/embed:Q8_0"))
        self.assertEqual(len(self.runs()), 1)
        with open(os.path.join(self.tmp.name, "embed.env")) as f:
            self.assertIn("ROLE=embed", f.read())

    def test_same_hash_keeps_it(self):
        s = base_settings()
        found = {"luna-embed": {"state": "running", "labels": {"luna.config-hash": self.wanted_hash("embed", s)}}}
        r = lifecycle.ensure("embed", s, self.ctx, found=found)
        self.assertEqual(r.state, "kept")
        self.assertEqual(self.runs(), [])

    def test_a_setting_change_replaces_it(self):
        s = base_settings()
        found = {"luna-embed": {"state": "running", "labels": {"luna.config-hash": self.wanted_hash("embed", s)}}}
        s["EMBED_CTX"] = "4096"
        r = lifecycle.ensure("embed", s, self.ctx, found=found)
        self.assertEqual(r.state, "started")
        self.assertIn("--replace", self.runs()[0])

    def test_a_stopped_one_is_made_again(self):
        s = base_settings()
        found = {"luna-embed": {"state": "exited", "labels": {"luna.config-hash": self.wanted_hash("embed", s)}}}
        self.assertEqual(lifecycle.ensure("embed", s, self.ctx, found=found).state, "started")

    def test_embed_on_cpu_image_rerank_on_gpu(self):
        lifecycle.ensure("embed", base_settings(), self.ctx, found={})
        lifecycle.ensure("rerank", base_settings(), self.ctx, found={})
        embed, rerank = self.runs()
        self.assertEqual(embed[-1], "IMGCPU")
        self.assertNotIn("nvidia.com/gpu=all", embed)
        self.assertEqual(rerank[-1], "IMGGPU")
        self.assertIn("nvidia.com/gpu=all", rerank)

    def test_no_image_yet_says_run_setup(self):
        self.ctx.image_ids = {}
        r = lifecycle.ensure("embed", base_settings(), self.ctx, found={})
        self.assertEqual(r.state, "skipped")
        self.assertIn("setup", r.note)

    def test_skipped_role(self):
        r = lifecycle.ensure("rerank", base_settings(RERANK="off"), self.ctx, found={})
        self.assertEqual(r.state, "skipped")
        self.assertIn("RERANK=off", r.note)

    def test_refused_chat(self):
        r = lifecycle.ensure("chat", base_settings(MODEL_HF="", MODEL_MATCH="*x*"), self.ctx, found={})
        self.assertEqual(r.state, "failed")
        self.assertIn("MODEL_HF", r.note)

    def test_podman_failing_is_one_line(self):
        def fail(cmd, **kw):
            return subprocess.CompletedProcess(cmd, 125, "", "Error: something\nError: the real reason")
        with mock.patch.object(lifecycle.subprocess, "run", side_effect=fail):
            r = lifecycle.ensure("embed", base_settings(), self.ctx, found={})
        self.assertEqual((r.state, r.note), ("failed", "Error: the real reason"))


class Ready(unittest.TestCase):
    def clock(self):
        t = [0.0]

        def now():
            return t[0]

        def pause(sec):
            t[0] += sec
        return now, pause

    def test_polls_until_ready(self):
        codes = iter([None, 503, 503, 200])

        def get(url, timeout):
            code = next(codes)
            if code is None:
                raise requests.ConnectionError()
            return mock.Mock(status_code=code)
        now, pause = self.clock()
        self.assertTrue(lifecycle.wait_ready("http://127.0.0.1:8080", 60, get=get, clock=now, pause=pause))
        self.assertEqual(now(), 3)

    def test_times_out(self):
        now, pause = self.clock()
        get = mock.Mock(return_value=mock.Mock(status_code=503))
        self.assertFalse(lifecycle.wait_ready("http://127.0.0.1:8080", 10, get=get, clock=now, pause=pause))
        self.assertEqual(now(), 10)


class Down(unittest.TestCase):
    def test_stops_only_ours(self):
        found = {"luna-llama": {"state": "running", "labels": {}},
                 "luna-embed": {"state": "running", "labels": {"luna.config-hash": "x"}}}
        with mock.patch.object(lifecycle, "podman") as podman:
            gone = lifecycle.down(found=found)
        self.assertEqual(gone, ["luna-embed"])
        podman.assert_called_once_with("rm", "-f", "-t", "10", "luna-embed")

    def test_only_the_roles_asked(self):
        found = {"luna-embed": {"state": "running", "labels": {"luna.config-hash": "x"}},
                 "luna-rerank": {"state": "running", "labels": {"luna.config-hash": "y"}}}
        with mock.patch.object(lifecycle, "podman"):
            self.assertEqual(lifecycle.down(("rerank",), found=found), ["luna-rerank"])


class Podman(unittest.TestCase):
    def test_missing_podman_says_how_to_get_it(self):
        with mock.patch.object(lifecycle.shutil, "which", return_value=None):
            self.assertIn("isn't installed", lifecycle.podman_problem())

    def test_too_old(self):
        with mock.patch.object(lifecycle.shutil, "which", return_value="/usr/bin/podman"), \
             mock.patch.object(lifecycle, "podman", return_value=mock.Mock(stdout="4.6.2\n")):
            self.assertIn("4.7.0", lifecycle.podman_problem())

    def test_not_rootless(self):
        answers = iter([mock.Mock(stdout="5.8.8\n"), mock.Mock(stdout="false\n")])
        with mock.patch.object(lifecycle.shutil, "which", return_value="/usr/bin/podman"), \
             mock.patch.object(lifecycle, "podman", side_effect=lambda *a, **k: next(answers)):
            self.assertIn("rootless", lifecycle.podman_problem())


if __name__ == "__main__":
    unittest.main()
