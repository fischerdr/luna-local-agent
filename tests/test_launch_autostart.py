"""The launcher with containers (launch.py, config.json "services"): with no
"services" key it does exactly what main's launcher did; with autostart it
brings them up first, starts nothing twice, and stops what it started.
Every server, container and process is pretended.

    python -m unittest discover tests        (from the repo folder)
"""
import contextlib
import importlib.util
import io
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.offline import FAKE  # noqa: E402,F401  (first: no real servers)

import launch  # noqa: E402
from services import lifecycle, settings as service_settings  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANAGED = {"voice": {"mode": "managed", "python": "/k/py", "server": "/k/s.py", "port": 8899}}
CONTAINER_VOICE = {"voice": {"mode": "external", "url": "http://127.0.0.1:8899", "container": "luna-tts"}}


def run_main(module, cfg, st, results=(), ready=True, model_up=False):
    """main() with everything outside pretended: the calls it made, in order."""
    calls = []

    def rec(name, value=None):
        def f(*a, **k):
            calls.append(name)
            return value
        return f

    patches = [
        mock.patch.object(module, "settings", return_value=cfg),
        mock.patch.object(module, "state", return_value=st),
        mock.patch.object(module, "seed_config", return_value=False),
        mock.patch.object(module, "python", return_value="/venv/python"),
        mock.patch.object(module, "ensure_packages", side_effect=rec("ensure_packages", True)),
        mock.patch.object(module, "start_llama", side_effect=rec("start_llama")),
        mock.patch.object(module, "start_voice", side_effect=rec("start_voice")),
        mock.patch.object(module, "stop_group", side_effect=rec("stop_group")),
        mock.patch.object(module, "model_up", return_value=model_up),
        mock.patch.object(module, "up", return_value=True),
        mock.patch.object(module.subprocess, "call", side_effect=rec("main.py", 0)),
        mock.patch.object(sys, "argv", ["launch.py"]),
        mock.patch.object(sys, "platform", "linux"),
        mock.patch("shutil.which", return_value="/usr/bin/podman"),
        mock.patch.object(service_settings, "load", return_value=dict(service_settings.DEFAULTS, READY_TIMEOUT="5")),
        mock.patch.object(lifecycle, "up", side_effect=lambda *a, **k: calls.append("services up") or list(results)),
        mock.patch.object(lifecycle, "default_roles", return_value=("chat", "embed", "rerank", "tts")),
        mock.patch.object(lifecycle, "wait_ready", side_effect=lambda *a, **k: calls.append("wait") or ready),
        mock.patch.object(lifecycle, "running", return_value=lambda: True),
        mock.patch.object(lifecycle, "down", side_effect=lambda roles: calls.append(("services down", tuple(roles)))),
    ]
    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
        with contextlib.suppress(SystemExit):
            module.main()
    return calls


def results(*pairs):
    return [lifecycle.Result(role, how, "") for role, how in pairs]


class NoServicesKey(unittest.TestCase):
    def test_nothing_new_happens(self):
        for st in (MANAGED, CONTAINER_VOICE, {}):
            with self.subTest(state=st):
                calls = run_main(launch, {}, st)
                self.assertNotIn("services up", calls)
                self.assertNotIn("wait", calls)
                self.assertEqual(calls[:2], ["ensure_packages", "start_llama"])

    def test_autostart_false_is_the_same(self):
        self.assertEqual(run_main(launch, {"services": {"autostart": False}}, MANAGED),
                         run_main(launch, {}, MANAGED))

    def test_same_calls_in_the_same_order_as_mains_launcher(self):
        base = subprocess.run(["git", "-C", ROOT, "merge-base", "HEAD", "main"], capture_output=True, text=True)
        rev = os.environ.get("FENCE_BASE") or base.stdout.strip()
        src = subprocess.run(["git", "-C", ROOT, "show", f"{rev}:launch.py"], capture_output=True, text=True) \
            if rev else None
        if not src or src.returncode:
            self.skipTest("main's launch.py isn't in this checkout's history")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "launch_main.py")
            with open(path, "w") as f:
                f.write(src.stdout)
            spec = importlib.util.spec_from_file_location("launch_main", path)
            main_launch = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(main_launch)
        for st in (MANAGED, CONTAINER_VOICE, {"voice": {"mode": "none"}}):
            with self.subTest(state=st):
                self.assertEqual(run_main(launch, {}, st), run_main(main_launch, {}, st))


class Autostart(unittest.TestCase):
    cfg = {"services": {"autostart": True}}

    def test_up_then_wait_then_mains_flow_then_down(self):
        calls = run_main(launch, self.cfg, CONTAINER_VOICE,
                         results(("chat", "started"), ("embed", "started"), ("rerank", "kept"), ("tts", "started")))
        self.assertEqual(calls[:6], ["ensure_packages", "services up", "wait", "wait", "wait", "wait"])
        self.assertLess(calls.index("main.py"), calls.index(("services down", ("chat", "embed", "tts"))))
        # rerank was running before: left alone, like a server main finds already up

    def test_chat_container_means_no_start_sh(self):
        for how in ("started", "kept"):
            with self.subTest(how=how):
                calls = run_main(launch, self.cfg, MANAGED, results(("chat", how)), ready=False)
                self.assertNotIn("start_llama", calls)

    def test_chat_that_didnt_come_falls_back_to_mains_flow(self):
        calls = run_main(launch, self.cfg, MANAGED, results(("chat", "skipped")))
        self.assertIn("start_llama", calls)

    def test_container_voice_means_no_host_voice(self):
        calls = run_main(launch, self.cfg, MANAGED, results(("chat", "started"), ("tts", "started")))
        self.assertNotIn("start_voice", calls)

    def test_host_voice_still_starts_without_a_voice_container(self):
        calls = run_main(launch, self.cfg, MANAGED, results(("chat", "started")))
        self.assertIn("start_voice", calls)

    def test_stop_on_exit_false_leaves_them_running(self):
        cfg = {"services": {"autostart": True, "stop_on_exit": False}}
        calls = run_main(launch, cfg, CONTAINER_VOICE, results(("chat", "started")))
        self.assertFalse([c for c in calls if isinstance(c, tuple)])

    def test_never_blocks_her_start(self):
        calls = run_main(launch, self.cfg, MANAGED, results(("chat", "failed"), ("embed", "skipped")), ready=False)
        self.assertIn("main.py", calls)
        self.assertIn("start_llama", calls)

    def test_an_exception_in_services_still_starts_her(self):
        calls = []
        with mock.patch.object(service_settings, "load", side_effect=service_settings.SettingsError("bad env")):
            with mock.patch.object(launch, "settings", return_value=self.cfg), \
                 mock.patch("shutil.which", return_value="/usr/bin/podman"), \
                 mock.patch.object(sys, "platform", "linux"), \
                 contextlib.redirect_stderr(io.StringIO()) as err:
                calls.append(launch.start_services())
        self.assertEqual(calls, [None])
        self.assertIn("bad env", err.getvalue())

    def test_no_podman(self):
        with mock.patch.object(launch, "settings", return_value=self.cfg), \
             mock.patch("shutil.which", return_value=None), \
             mock.patch.object(sys, "platform", "linux"), \
             contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertIsNone(launch.start_services())
        self.assertIn("Podman isn't installed", err.getvalue())

    def test_not_on_windows_or_macos(self):
        for platform in ("win32", "darwin"):
            with mock.patch.object(launch, "settings", return_value=self.cfg), \
                 mock.patch.object(sys, "platform", platform):
                self.assertIsNone(launch.start_services())


if __name__ == "__main__":
    unittest.main()
