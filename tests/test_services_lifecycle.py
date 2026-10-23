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


class Voice(Base):
    def setUp(self):
        super().setUp()
        self.ctx.image_ids.update({("tts", "cpu"): "TTSCPU", ("tts", "cuda13"): "TTSGPU"})
        p = mock.patch.object(lifecycle, "config_voice", return_value="luna")
        p.start()
        self.addCleanup(p.stop)

    def test_never_while_the_host_voice_is_managed(self):
        r = lifecycle.ensure_tts(base_settings(), self.ctx, found={}, voice_mode="managed")
        self.assertEqual(r.state, "skipped")
        self.assertIn("setup tts", r.note)
        self.assertEqual(self.runs(), [])

    def test_starts_with_the_configured_voice(self):
        r = lifecycle.ensure_tts(base_settings(), self.ctx, found={}, voice_mode="external")
        self.assertEqual(r.state, "started")
        with open(os.path.join(self.tmp.name, "tts.env")) as f:
            self.assertIn("KOKORO_VOICE=luna", f.read())

    def test_gpu_on_nvidia_unless_told_cpu(self):
        lifecycle.ensure_tts(base_settings(), self.ctx, found={}, voice_mode="external")
        self.assertEqual(self.runs()[-1][-1], "TTSGPU")
        lifecycle.ensure_tts(base_settings(TTS_DEVICE="cpu"), self.ctx, found={}, voice_mode="external")
        self.assertEqual(self.runs()[-1][-1], "TTSCPU")
        self.ctx.backend = "rocm"           # no ROCm torch: an AMD machine's voice is on its CPU
        lifecycle.ensure_tts(base_settings(), self.ctx, found={}, voice_mode="external")
        self.assertEqual(self.runs()[-1][-1], "TTSCPU")

    def test_default_roles_take_the_voice_once_it_is_a_container(self):
        with mock.patch.object(lifecycle, "voice_state", return_value={"mode": "managed"}):
            self.assertEqual(lifecycle.default_roles(), ("chat", "embed", "rerank"))
        with mock.patch.object(lifecycle, "voice_state", return_value={"mode": "external", "container": "luna-tts"}):
            self.assertEqual(lifecycle.default_roles(), ("chat", "embed", "rerank", "tts"))


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

    def test_stops_waiting_when_the_container_stopped(self):
        now, pause = self.clock()
        get = mock.Mock(side_effect=requests.ConnectionError())
        self.assertFalse(lifecycle.wait_ready("http://127.0.0.1:8080", 600, get=get, clock=now, pause=pause,
                                              alive=lambda: now() < 3))
        self.assertEqual(now(), 3)


class Trouble(unittest.TestCase):
    def test_gated_model_says_how_to_log_in(self):
        log = ("0.00.137.429 E get_repo_commit: error: GET failed (401): Access to model "
               "google/gemma-3-1b-it-qat-q4_0-gguf is restricted. You must have access to it and be "
               "authenticated to access it. Please log in.\n")
        hint = lifecycle.trouble("chat", log=log)
        self.assertIn("google/gemma-3-1b-it-qat-q4_0-gguf is gated", hint)
        self.assertIn("huggingface-cli login", hint)
        self.assertIn("setup", hint)

    def test_no_such_model(self):
        self.assertIn("*_HF", lifecycle.trouble("embed", log="E get_repo_commit: error: GET failed (404): x"))

    def test_out_of_memory(self):
        self.assertIn("NGL", lifecycle.trouble("chat", log="ggml_backend_cuda_buffer: cudaMalloc failed: out of memory"))

    def test_nothing_known(self):
        self.assertIsNone(lifecycle.trouble("chat", log="srv load_model: loading model\n"))


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


class Reset(unittest.TestCase):
    PS = [{"Names": ["luna-llama"], "State": "running", "Labels": {"luna.config-hash": "a"}},
          {"Names": ["luna-tts"], "State": "running", "Labels": {"luna.config-hash": "b"}},
          {"Names": ["luna-old"], "State": "exited", "Labels": {}},
          {"Names": ["luna-test-chat"], "State": "running", "Labels": {}},
          {"Names": ["nextcloud"], "State": "running", "Labels": {}}]
    IMAGES = [{"Id": "mine", "Names": ["localhost/luna-services-llama:cpu"], "Labels": {"luna.kind": "llama"}},
              {"Id": "deploy", "Names": ["localhost/luna-deploy-tts:cpu"], "Labels": {}},
              {"Id": "old", "Names": ["localhost/luna-llama:cuda13"], "Labels": {"maintainer": "NVIDIA"}},
              {"Id": "test", "Names": ["localhost/luna-test-llama:cpu"], "Labels": {}},
              {"Id": "base", "Names": ["ghcr.io/ggml-org/llama.cpp:server"], "Labels": {}}]
    VOLS = ["luna-models", "luna-data", "luna-tts-cache", "luna-test-models", "other"]
    SECRETS = ["luna_api_key", "luna_hf_token", "luna_test_api_key", "someone_elses"]

    def plan(self):
        return lifecycle.reset_plan(self.PS, self.IMAGES, self.VOLS, self.SECRETS)

    def test_only_luna_things(self):
        p = self.plan()
        self.assertEqual(p["containers"], ["luna-llama", "luna-tts"])
        self.assertEqual(sorted(p["images"]), ["deploy", "mine"])
        self.assertEqual(p["volumes"], ["luna-data", "luna-tts-cache"])
        self.assertEqual(p["secrets"], ["luna_api_key", "luna_hf_token"])
        self.assertEqual(p["models"], ["luna-models"])
        self.assertEqual((p["strangers"], p["strange_images"]), (["luna-old"], ["old"]))

    def run_reset(self, answers):
        asked = []

        def ask(question, default=True):
            asked.append(question)
            return answers.pop(0)
        with mock.patch.object(lifecycle, "podman") as podman:
            gone = lifecycle.reset(ask, out=lambda *a: None, plan=self.plan())
        return gone, asked, [c.args for c in podman.call_args_list]

    def test_asks_before_luna_models_and_keeps_it_by_default(self):
        gone, asked, calls = self.run_reset([True, False, False])
        self.assertIn("luna-models", asked[1])
        self.assertNotIn("luna-models", gone)
        self.assertNotIn(("volume", "rm", "-f", "luna-models"), calls)
        self.assertIn(("volume", "rm", "-f", "luna-data"), calls)
        self.assertNotIn("luna-old", gone)

    def test_no_means_nothing_removed(self):
        gone, _, calls = self.run_reset([False, False, False])
        self.assertEqual((gone, calls), ([], []))

    def test_never_test_or_other_peoples_things(self):
        _, _, calls = self.run_reset([True, True, True])
        flat = " ".join(" ".join(c) for c in calls)
        for name in ("luna-test", "nextcloud", "other", "someone_elses", "base"):
            self.assertNotIn(name, flat)


class Status(unittest.TestCase):
    def lines(self, s, voice):
        with mock.patch.object(lifecycle, "voice_state", return_value=voice):
            return dict((ln.split()[0], (good, ln)) for good, ln in
                        lifecycle.status(s, found={}, get=mock.Mock(side_effect=requests.ConnectionError())))

    def test_a_role_switched_off_is_fine(self):
        out = self.lines(base_settings(RERANK="off"), {"container": "luna-tts"})
        self.assertEqual(out["rerank"][0], True)
        self.assertIn("RERANK=off", out["rerank"][1])
        self.assertEqual(out["chat"][0], False)                  # not created, and should be

    def test_a_host_voice_is_fine(self):
        out = self.lines(base_settings(), {"mode": "managed"})
        self.assertEqual(out["tts"][0], True)
        self.assertEqual(self.lines(base_settings(), {"container": "luna-tts"})["tts"][0], False)


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
