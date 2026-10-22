"""When the installer suggests the voice container (installer/install.py,
voice()): only where main's default would fail. Every outside thing -
Podman, pip, git, the network - is pretended.

    python -m unittest discover tests        (from the repo folder)
"""
import contextlib
import importlib.util
import io
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.offline import FAKE  # noqa: E402,F401  (first: no real servers)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("luna_installer_voice", os.path.join(ROOT, "installer", "install.py"))
install = importlib.util.module_from_spec(spec)
spec.loader.exec_module(install)


class VoiceChoice(unittest.TestCase):
    def go(self, *argv, podman=True, python_ok=True, answering=False, pip_ok=True, asks=None, state=None):
        """Run voice() and say what it did: (container?, host install?, output, state)."""
        install.ARGS = install.parser().parse_args(list(argv))
        install.REPORT.clear()
        state = {"kokoro_dir": "/k"} if state is None else state
        patches = [
            mock.patch.object(install, "SYSTEM", "Linux"),
            mock.patch.object(install.shutil, "which", side_effect=lambda n: "/usr/bin/podman" if podman and n == "podman"
                              else "/usr/bin/" + n if n != "podman" else None),
            mock.patch.object(install, "kokoro_python_ok", return_value=python_ok),
            mock.patch.object(install, "http_ok", return_value=answering),
            mock.patch.object(install, "voice_container", return_value=True),
            mock.patch.object(install, "run", return_value=True),
            mock.patch.object(install, "pip", return_value=pip_ok),
            mock.patch.object(install, "set_config"),
            mock.patch.object(install.os.path, "isdir", return_value=True),
            mock.patch.object(install.os.path, "exists", return_value=True),
            mock.patch.object(install.subprocess, "run", return_value=mock.Mock(returncode=0)),
        ]
        if asks is not None:
            patches.append(mock.patch.object(install, "ask", side_effect=asks))
        with contextlib.ExitStack() as stack:
            mocks = [stack.enter_context(p) for p in patches]
            out = stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            install.voice(state, "cpu")
        container = mocks[4]
        return container.called, state.get("voice", {}).get("mode") == "managed", out.getvalue(), state

    def test_explicit_voice_is_done_as_given(self):
        container, host, _, _ = self.go("--voice", "install", python_ok=False)
        self.assertEqual((container, host), (False, True))
        container, _, _, _ = self.go("--voice", "container")
        self.assertTrue(container)

    def test_explicit_install_gets_no_container_offer_when_pip_fails(self):
        offered = mock.Mock(return_value=True)
        container, _, _, _ = self.go("--voice", "install", pip_ok=False, asks=offered)
        self.assertFalse(container)
        offered.assert_not_called()

    def test_something_on_8899_is_used(self):
        # (not one the installer made: that one has a kokoro_dir)
        container, host, _, state = self.go(answering=True, python_ok=False, state={})
        self.assertFalse(container)
        self.assertEqual(state["voice"], {"mode": "external", "url": "http://127.0.0.1:8899"})

    def test_python_outside_kokoros_range_defaults_to_the_container(self):
        container, _, out, _ = self.go("--yes", python_ok=False)
        self.assertTrue(container)
        self.assertIn("Kokoro needs Python 3.10-3.12", out)

    def test_but_it_still_asks(self):
        with mock.patch("builtins.input", return_value="2"):
            container, host, _, _ = self.go(python_ok=False)
        self.assertEqual((container, host), (False, True))

    def test_otherwise_the_host_venv_as_on_main(self):
        container, host, _, _ = self.go("--yes")
        self.assertEqual((container, host), (False, True))

    def test_pip_failing_offers_the_container(self):
        container, _, out, _ = self.go(pip_ok=False, asks=[True])
        self.assertTrue(container)
        self.assertIn("didn't install", out)

    def test_no_podman_is_mains_behaviour_plus_a_hint(self):
        container, host, out, _ = self.go(podman=False, python_ok=False, pip_ok=False)
        self.assertFalse(container)
        self.assertTrue(host)            # main's managed state, failed packages and all
        self.assertIn("voice server packages failed", out)
        self.assertIn("--voice container", out)


class NewerPython(unittest.TestCase):
    """With only a Python newer than 3.13 (install.sh goes on with it when
    Podman is there), the model server defaults to containers too."""

    def pick(self, version, podman, *argv):
        install.ARGS = install.parser().parse_args(list(argv))
        state = {}
        with mock.patch.object(install, "SYSTEM", "Linux"), \
             mock.patch.object(install.sys, "version_info", version), \
             mock.patch.object(install.shutil, "which", side_effect=lambda n: "/usr/bin/podman"
                               if podman and n == "podman" else None), \
             mock.patch.object(install, "http_ok", return_value=False), \
             mock.patch("builtins.input", return_value=""), \
             contextlib.redirect_stdout(io.StringIO()):
            install.backend(state)
        return state["backend"]

    def test_newer_python_and_podman(self):
        self.assertEqual(self.pick((3, 14, 0), True, "--yes"), "container")
        self.assertEqual(self.pick((3, 14, 0), True), "container")          # Enter takes it

    def test_otherwise_mains_default(self):
        self.assertEqual(self.pick((3, 12, 4), True, "--yes"), "lmstudio")
        self.assertEqual(self.pick((3, 14, 0), False, "--yes"), "lmstudio")

    def test_an_explicit_backend_wins(self):
        with mock.patch.object(install, "run", return_value=True), \
             mock.patch.object(install, "detect_gpu", return_value=("cuda", "NVIDIA")):
            self.assertEqual(self.pick((3, 14, 0), True, "--backend", "llama"), "llama")

    def test_the_rule(self):
        with mock.patch.object(install, "SYSTEM", "Linux"), \
             mock.patch.object(install.shutil, "which", return_value="/usr/bin/podman"):
            self.assertTrue(install.python_wants_containers((3, 14)))
            self.assertFalse(install.python_wants_containers((3, 13)))
        with mock.patch.object(install, "SYSTEM", "Darwin"), \
             mock.patch.object(install.shutil, "which", return_value="/usr/bin/podman"):
            self.assertFalse(install.python_wants_containers((3, 14)))


class KokoroRange(unittest.TestCase):
    def test_range(self):
        self.assertTrue(install.kokoro_python_ok((3, 10)))
        self.assertTrue(install.kokoro_python_ok((3, 12)))
        self.assertFalse(install.kokoro_python_ok((3, 13)))
        self.assertFalse(install.kokoro_python_ok((3, 9)))


if __name__ == "__main__":
    unittest.main()
