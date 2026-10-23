"""python -m backends.containers' own lifecycle against real Podman: a second up
keeps everything, a settings edit makes only that role anew, one role
stopping leaves the others answering, and down and reset leave nothing.

Its names are swapped for luna-test-* ones (containers, volumes, secrets),
its settings files are temporary ones with the test ports, and its key is
a test key - your own containers, models and llama/.api_key are untouched.
"""
import os
import shutil
import tempfile
import unittest
from unittest import mock

import requests

from backends.containers import images, keys, lifecycle, run, settings
from backends.containers.tests.live import support
from backends.containers.tests.live.support import live, podman

NAMES = {"chat": "luna-test-chat", "embed": "luna-test-embed", "rerank": "luna-test-rerank", "tts": "luna-test-tts"}
VOLS = {"models": support.MODELS, "data": "luna-test-data", "tts": support.TTS_CACHE}


@live
class Lifecycle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.llama = os.path.join(cls.tmp.name, "llama")
        cls.services = os.path.join(cls.tmp.name, "services")
        os.makedirs(cls.llama)
        os.makedirs(cls.services)
        shutil.copy(os.path.join(support.ROOT, "llama", "server.env"), cls.llama)
        shutil.copy(os.path.join(support.ROOT, "backends", "containers", "services.env"), cls.services)
        with open(os.path.join(cls.llama, "server.local.env"), "w") as f:
            f.write("PORT=18080\nEMBED_PORT=18081\nRERANK_PORT=18082\nCTX=4096\nPARALLEL=1\nNGL=0\n")
        m = support.models()
        cls.write_local(f'MODEL_HF="{m["TEST_CHAT_HF"]}"\nEMBED_HF="{m["TEST_EMBED_HF"]}"\n'
                        f'RERANK_HF="{m["TEST_RERANK_HF"]}"\nRERANK_NGL=0\nLLAMA_BACKEND=cpu\n')
        cls.key = support.secrets.token_urlsafe(32)
        cls.patches = [
            mock.patch.dict("services.CONTAINER", NAMES),
            mock.patch.dict("services.VOLUME", VOLS),
            mock.patch.object(keys, "KEY_SECRET", "luna_test_api_key"),
            mock.patch.object(keys, "TOKEN_SECRET", "luna_test_hf_token"),
            mock.patch.object(run, "KEY_SECRET", "luna_test_api_key"),
            mock.patch.object(run, "TOKEN_SECRET", "luna_test_hf_token"),
            mock.patch.object(keys, "read_key", return_value=cls.key),
            mock.patch.object(lifecycle, "voice_state", return_value={"mode": "none"}),
        ]
        for p in cls.patches:
            p.start()
        if not images.find("cpu"):
            support.subprocess.run(images.build_cmd("cpu"), check=True, capture_output=True)

    @classmethod
    def write_local(cls, text):
        with open(os.path.join(cls.services, "services.local.env"), "w") as f:
            f.write(text)

    @classmethod
    def tearDownClass(cls):
        lifecycle.down(tuple(NAMES))
        for p in reversed(cls.patches):
            p.stop()
        podman("rm", "-f", *NAMES.values(), check=False)
        podman("volume", "rm", "-f", "luna-test-data", check=False)
        podman("secret", "rm", "luna_test_api_key", check=False)
        cls.tmp.cleanup()

    def settings(self):
        return settings.load(self.llama, self.services)

    def up(self):
        results = {r.role: r.state for r in lifecycle.up(self.settings(), ("chat", "embed", "rerank"))}
        for role in results:
            if results[role] in ("started", "kept"):
                self.assertTrue(lifecycle.wait_ready(support.url(role), support.READY,
                                                     alive=lifecycle.running(role)), role)
        return results

    def ids(self):
        return {role: podman("inspect", "-f", "{{.Id}}", name, check=False).stdout.strip()
                for role, name in NAMES.items() if role != "tts"}

    def test_lifecycle(self):
        # Run as one story: each step builds on the one before.
        self.assertEqual(self.up(), {"chat": "started", "embed": "started", "rerank": "started"})
        first = self.ids()

        self.assertEqual(self.up(), {"chat": "kept", "embed": "kept", "rerank": "kept"})
        self.assertEqual(self.ids(), first)

        # A setting of the reranker's: only it is made anew.
        with open(os.path.join(self.llama, "server.local.env"), "a") as f:
            f.write("RERANK_CTX=1024\n")
        self.assertEqual(self.up(), {"chat": "kept", "embed": "kept", "rerank": "started"})
        now = self.ids()
        self.assertEqual((now["chat"], now["embed"]), (first["chat"], first["embed"]))
        self.assertNotEqual(now["rerank"], first["rerank"])

        # The embedder stopping leaves chat answering.
        podman("stop", "-t", "2", NAMES["embed"])
        self.assertEqual(requests.get(support.url("chat", "/health"), timeout=5).status_code, 200)

        # down, then reset (without touching images, which are shared with your own setup).
        self.assertEqual(sorted(lifecycle.down(("chat", "embed", "rerank"))),
                         sorted(NAMES[r] for r in ("chat", "embed", "rerank")))
        # Your own luna-* containers carry the same label as these: the plan
        # is cut down to the test's, and they must all still be there after.
        yours = sorted(n for n in podman("ps", "-a", "--format", "{{.Names}}").stdout.split()
                       if n not in NAMES.values())
        plan = lifecycle.reset_plan()
        plan.update(containers=[n for n in plan["containers"] if n in NAMES.values()],
                    images=[], strange_images=[], strangers=[])
        lifecycle.reset(lambda q, default=True: default, out=lambda *a: None, plan=plan)
        left = podman("ps", "-a", "--format", "{{.Names}}").stdout.split()
        self.assertEqual(sorted(n for n in left if n not in NAMES.values()), yours)
        self.assertFalse([n for n in left if n in NAMES.values()])
        self.assertNotIn("luna_test_api_key", podman("secret", "ls", "--format", "{{.Name}}").stdout.split())
        vols = podman("volume", "ls", "--format", "{{.Name}}").stdout.split()
        self.assertNotIn("luna-test-data", vols)
        self.assertIn(support.MODELS, vols)            # the models stay unless you say otherwise


if __name__ == "__main__":
    unittest.main()
