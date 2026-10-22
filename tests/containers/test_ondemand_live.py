"""The embedder on demand, for real: wanted while it's down, its container
comes up in the background; idle, the reaper stops it. Under luna-test-*
names, with temporary settings on the test ports, like test_lifecycle_live.
"""
import os
import shutil
import tempfile
import time
import unittest
from unittest import mock

from services import keys, lifecycle, ondemand, run, settings
from tests.containers import support
from tests.containers.support import live, podman
from tests.containers.test_lifecycle_live import NAMES, VOLS


@live
class OnDemandLive(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        llama, services = os.path.join(self.tmp.name, "llama"), os.path.join(self.tmp.name, "services")
        os.makedirs(llama)
        os.makedirs(services)
        shutil.copy(os.path.join(support.ROOT, "llama", "server.env"), llama)
        shutil.copy(os.path.join(support.ROOT, "services", "services.env"), services)
        with open(os.path.join(llama, "server.local.env"), "w") as f:
            f.write("EMBED_PORT=18081\n")
        with open(os.path.join(services, "services.local.env"), "w") as f:
            f.write(f'EMBED_HF="{support.models()["TEST_EMBED_HF"]}"\nLLAMA_BACKEND=cpu\n'
                    "EMBED_ONDEMAND=on\nEMBED_IDLE=5\n")
        s = settings.load(llama, services)
        key = support.secrets.token_urlsafe(32)
        for p in (mock.patch.dict("services.CONTAINER", NAMES), mock.patch.dict("services.VOLUME", VOLS),
                  mock.patch.object(keys, "KEY_SECRET", "luna_test_api_key"),
                  mock.patch.object(keys, "TOKEN_SECRET", "luna_test_hf_token"),
                  mock.patch.object(run, "KEY_SECRET", "luna_test_api_key"),
                  mock.patch.object(run, "TOKEN_SECRET", "luna_test_hf_token"),
                  mock.patch.object(keys, "read_key", return_value=key),
                  mock.patch.object(keys, "token_source", return_value=("", "")),
                  mock.patch.object(ondemand, "_settings", s),
                  mock.patch.object(ondemand, "_ensure_reaper")):
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(podman, "secret", "rm", "luna_test_api_key", check=False)
        self.addCleanup(podman, "rm", "-f", NAMES["embed"], check=False)
        ondemand._started.clear()
        ondemand._starting.clear()

    def test_started_when_wanted_and_stopped_when_idle(self):
        url = support.url("embed", "/v1/embeddings")
        t = time.monotonic()
        ondemand.wanted("embed", False, url)
        self.assertLess(time.monotonic() - t, 0.5)              # the turn went on
        self.assertTrue(lifecycle.wait_ready(support.url("embed"), support.READY), "never came up")
        for _ in range(100):                                   # the thread notes it started it
            if "embed" in ondemand._started:
                break
            time.sleep(0.1)
        self.assertIn("embed", ondemand._started)

        self.assertEqual(ondemand.reap(clock=lambda: time.monotonic()), [])        # just wanted
        self.assertEqual(ondemand.reap(clock=lambda: time.monotonic() + 10), ["embed"])
        left = podman("ps", "-a", "--format", "{{.Names}}").stdout.split()
        self.assertNotIn(NAMES["embed"], left)


if __name__ == "__main__":
    unittest.main()
