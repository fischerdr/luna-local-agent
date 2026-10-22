"""Knowing what's ours (services/lifecycle.py, services/images.py): an
image by its labels, never its tag; a luna-* container someone else made
is listed for setup to ask about, never replaced; a busy port is named
and left alone. Podman and ss are pretended.

    python -m unittest discover tests        (from the repo folder)
"""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.offline import FAKE  # noqa: E402,F401  (first: no real servers)

from services import images, lifecycle  # noqa: E402
from tests.test_services_lifecycle import Base, base_settings  # noqa: E402


class NotOurs(Base):
    def test_unlabelled_container_is_never_replaced(self):
        found = lifecycle.containers([{"Names": ["luna-llama"], "State": "running", "Labels": {"maintainer": "x"}}])
        r = lifecycle.ensure("chat", base_settings(), self.ctx, found=found)
        self.assertEqual(r.state, "skipped")
        self.assertIn("didn't make", r.note)
        self.assertEqual(self.runs(), [])

    def test_unlabelled_listed_for_the_migration_prompt(self):
        found = lifecycle.containers([
            {"Names": ["luna-llama"], "State": "running", "Labels": {}},
            {"Names": ["luna-tts"], "State": "exited", "Labels": None},
            {"Names": ["luna-embed"], "State": "running", "Labels": {"luna.config-hash": "x"}},
            {"Names": ["luna-test-chat"], "State": "running", "Labels": {}},
            {"Names": ["someone-else"], "State": "running", "Labels": {}},
        ])
        self.assertEqual(lifecycle.unlabelled(found), ["luna-llama", "luna-tts"])

    def test_tag_alone_is_not_our_image(self):
        retag = {"Id": "old", "Names": ["localhost/luna-llama:cuda13"], "Labels": {"maintainer": "NVIDIA"}}
        self.assertIsNone(images.find("cuda13", [retag]))
        ours = {"Id": "new", "Labels": images.labels("cuda13")}
        self.assertEqual(images.find("cuda13", [retag, ours]), "new")

    def test_stale_labels_are_not_current(self):
        stale = {"Id": "stale", "Labels": dict(images.labels("cpu"), **{"luna.image-version": "0"})}
        self.assertIsNone(images.find("cpu", [stale]))


class Ports(Base):
    def test_busy_port_names_the_holder_and_skips(self):
        with mock.patch.object(lifecycle, "port_holder", return_value="llama-server (pid 42)"):
            r = lifecycle.ensure("chat", base_settings(), self.ctx, found={})
        self.assertEqual(r.state, "skipped")
        self.assertIn("llama-server (pid 42)", r.note)
        self.assertEqual(self.runs(), [])

    def test_our_running_container_holding_it_is_fine(self):
        found = {"luna-llama": {"state": "running", "labels": {"luna.config-hash": "old"}}}
        with mock.patch.object(lifecycle, "port_holder", return_value="rootlessport (pid 9)"):
            r = lifecycle.ensure("chat", base_settings(), self.ctx, found=found)
        self.assertEqual(r.state, "started")


class ReadingSs(unittest.TestCase):
    def test_reading_ss(self):
        ss = 'LISTEN 0 512 0.0.0.0:8080 0.0.0.0:* users:(("llama-server",pid=461896,fd=16))\n'
        self.assertEqual(lifecycle.port_holder(8080, ss=ss), "llama-server (pid 461896)")
        self.assertIsNone(lifecycle.port_holder(8080, ss=""))
        self.assertEqual(lifecycle.port_holder(8080, ss="LISTEN 0 4096 127.0.0.1:8080 0.0.0.0:*\n"),
                         "another program")


if __name__ == "__main__":
    unittest.main()
