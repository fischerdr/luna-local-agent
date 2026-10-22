"""The embedder and reranker on demand (services/ondemand.py, and its hook
in embedmem.py): off, nothing changes; on, a server that's down is started
in the background while the turn goes on, and an idle one it started is
stopped again. Podman and every server are pretended.

    python -m unittest discover tests        (from the repo folder)
"""
import os
import sys
import threading
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.offline import FAKE  # noqa: E402,F401  (first: no real servers)

import config  # noqa: E402
import embedmem  # noqa: E402
from services import lifecycle, ondemand  # noqa: E402

LOCAL = "http://127.0.0.1:8081/v1/embeddings"


class Hook(unittest.TestCase):
    """embedmem's side."""

    def setUp(self):
        # embedmem's cache is shared with every other test: put it back after.
        for name in ("_state", "_rerank"):
            saved = dict(getattr(embedmem, name))
            self.addCleanup(lambda n=name, v=saved: (getattr(embedmem, n).clear(), getattr(embedmem, n).update(v)))
        embedmem._ondemand = None
        embedmem._state.update(up=None, checked=0.0)
        embedmem._rerank.update(up=None, checked=0.0)
        self.addCleanup(setattr, embedmem, "_ondemand", None)

    def test_no_services_block_changes_nothing(self):
        with mock.patch.dict(config.settings, {}, clear=False), \
             mock.patch.object(ondemand, "enabled") as enabled, \
             mock.patch.object(ondemand, "wanted") as wanted:
            config.settings.pop("services", None)
            self.assertFalse(embedmem.available())         # tests stay offline: not answering
            self.assertFalse(embedmem.reranker_available())
            self.assertFalse(embedmem.available())         # and again, from the cache
        enabled.assert_not_called()
        wanted.assert_not_called()
        self.assertIs(embedmem._ondemand, False)

    def test_switched_off_changes_nothing(self):
        with mock.patch.dict(config.settings, {"services": {"autostart": True}}), \
             mock.patch.object(ondemand, "enabled", return_value=False), \
             mock.patch.object(ondemand, "wanted") as wanted:
            embedmem.available()
            embedmem.reranker_available()
        wanted.assert_not_called()

    def test_on_it_is_told_each_time(self):
        with mock.patch.dict(config.settings, {"services": {"autostart": True}}), \
             mock.patch.object(ondemand, "enabled", return_value=True), \
             mock.patch.object(ondemand, "wanted") as wanted:
            self.assertFalse(embedmem.available())
            self.assertFalse(embedmem.available())          # the cached answer is still news
            self.assertFalse(embedmem.reranker_available())
        roles = [c.args[0] for c in wanted.call_args_list]
        self.assertEqual(roles, ["embed", "embed", "rerank"])
        self.assertEqual(wanted.call_args_list[0].args[1:], (False, config.EMBED_URL))

    def test_a_failing_lookup_is_off(self):
        with mock.patch.dict(config.settings, {"services": {"autostart": True}}), \
             mock.patch.object(ondemand, "enabled", side_effect=RuntimeError("no")):
            self.assertFalse(embedmem.available())
        self.assertIs(embedmem._ondemand, False)


class OnDemand(unittest.TestCase):
    def setUp(self):
        for name in ("_starting", "_started"):
            getattr(ondemand, name).clear()
        ondemand._last.clear()
        p = mock.patch.object(ondemand, "_ensure_reaper")
        p.start()
        self.addCleanup(p.stop)
        p = mock.patch.object(ondemand, "_load", return_value={"EMBED_IDLE": "600", "RERANK_IDLE": "60"})
        p.start()
        self.addCleanup(p.stop)
        p = mock.patch.object(ondemand.keys, "token_source", return_value=("", ""))
        p.start()
        self.addCleanup(p.stop)

    def test_starts_in_the_background_once(self):
        release, calls = threading.Event(), []

        def slow_up(s, roles, token, out=None):
            calls.append(roles)
            release.wait(5)
            return [lifecycle.Result(roles[0], "started")]
        with mock.patch.object(lifecycle, "up", side_effect=slow_up):
            t = time.monotonic()
            ondemand.wanted("embed", False, LOCAL)
            ondemand.wanted("embed", False, LOCAL)         # still starting: not twice
            self.assertLess(time.monotonic() - t, 0.5)     # the turn didn't wait
            release.set()
            for _ in range(50):
                if not ondemand._starting:
                    break
                time.sleep(0.02)
        self.assertEqual(calls, [("embed",)])
        self.assertEqual(ondemand._started, {"embed"})

    def test_up_does_nothing(self):
        with mock.patch.object(lifecycle, "up") as up:
            ondemand.wanted("embed", True, LOCAL)
        up.assert_not_called()

    def test_only_this_machines_servers(self):
        with mock.patch.object(lifecycle, "up") as up:
            ondemand.wanted("embed", False, "http://192.168.1.50:8081/v1/embeddings")
            ondemand.wanted("chat", False, "http://127.0.0.1:8080/v1/chat/completions")
        up.assert_not_called()

    def test_a_failed_start_is_tried_again_next_time(self):
        with mock.patch.object(lifecycle, "up", side_effect=RuntimeError("podman")):
            ondemand._start("embed")
        self.assertEqual((ondemand._starting, ondemand._started), (set(), set()))

    def test_reaper_stops_only_what_it_started_when_idle(self):
        ondemand._started.update({"embed", "rerank"})
        ondemand._last.update({"embed": 1000.0, "rerank": 1000.0})
        with mock.patch.object(lifecycle, "down") as down:
            self.assertEqual(ondemand.reap(clock=lambda: 1030.0), [])
            self.assertEqual(ondemand.reap(clock=lambda: 1061.0), ["rerank"])     # RERANK_IDLE=60
            self.assertEqual(ondemand.reap(clock=lambda: 1601.0), ["embed"])      # EMBED_IDLE=600
            self.assertEqual(ondemand.reap(clock=lambda: 9999.0), [])
        self.assertEqual([c.args[0] for c in down.call_args_list], [("rerank",), ("embed",)])

    def test_wanted_keeps_it_alive(self):
        ondemand._started.add("embed")
        ondemand.wanted("embed", True, LOCAL, clock=lambda: 2000.0)
        with mock.patch.object(lifecycle, "down") as down:
            self.assertEqual(ondemand.reap(clock=lambda: 2500.0), [])
        down.assert_not_called()


class NeverInATestRun(unittest.TestCase):
    def test_tests_never_start_real_containers(self):
        with mock.patch.object(ondemand, "_load", return_value={"EMBED_ONDEMAND": "on"}), \
             mock.patch.object(ondemand.shutil, "which", return_value="/usr/bin/podman"), \
             mock.patch.object(sys, "platform", "linux"):
            self.assertFalse(ondemand.enabled())
            with mock.patch.dict(sys.modules, {"tests.offline": None}):
                del sys.modules["tests.offline"]
                self.assertTrue(ondemand.enabled())


class Roles(unittest.TestCase):
    def test_autostart_leaves_the_side_models_to_demand(self):
        with mock.patch.object(lifecycle, "voice_state", return_value={}):
            self.assertEqual(lifecycle.default_roles({"EMBED_ONDEMAND": "on"}), ("chat",))
            self.assertEqual(lifecycle.default_roles({"EMBED_ONDEMAND": "off"}), ("chat", "embed", "rerank"))
            self.assertEqual(lifecycle.default_roles(), ("chat", "embed", "rerank"))


if __name__ == "__main__":
    unittest.main()
