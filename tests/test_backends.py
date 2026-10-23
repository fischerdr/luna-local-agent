"""backends/: other places her model and voice can run, found by folder.
With none, the installer and the launcher do what they always did; with
one, its hooks are called at the right moments. A pretend backend in a
temporary folder - nothing is installed or started.

    python -m unittest discover tests        (from the repo folder)
"""
import importlib.util
import io
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tests.offline import FAKE  # noqa: E402,F401  (first: no real servers)

import backends  # noqa: E402
import launch  # noqa: E402

spec = importlib.util.spec_from_file_location("luna_installer_backends", os.path.join(ROOT, "installer", "install.py"))
install = importlib.util.module_from_spec(spec)
spec.loader.exec_module(install)

# A backend with every hook, recording what it was called with.
PRETEND = '''
CALLS = []
LOCAL = True


def add_args(ap):
    ap.add_argument("--pretend-addr")


def validate(inst, args):
    CALLS.append(("validate", args.backend))
    return (["no address"] if args.backend == "elsewhere" and not args.pretend_addr else []), ["a note"]


def backends(inst):
    return [{"key": "boxes", "label": "in boxes", "menu": True, "default": True, "local_model": LOCAL},
            {"key": "elsewhere", "label": "far away", "menu": False, "local_model": False}]


def backend(inst, state, key):
    CALLS.append(("backend", key))


def gpu_reserve(inst, state):
    return 1.0 if state.get("backend") == "boxes" else None


def model(inst, state, pick):
    CALLS.append(("model", pick and pick["key"]))


def voices(inst):
    return [{"key": "boxed", "label": "in a box", "default": True}]


def voice(inst, state, key):
    CALLS.append(("voice", key))
    state["voice"] = {"mode": "external", "url": "http://127.0.0.1:8899"}


def install(inst, state):
    CALLS.append(("install", inst.ARGS.backend))


def health(inst, state):
    CALLS.append(("health",))


def start(settings):
    CALLS.append(("start",))
    return {"handled": {"chat"}, "stop": lambda: CALLS.append(("stop",))}
'''


class WithBackends(unittest.TestCase):
    """backends/ pointed at a temporary folder holding the given ones."""
    folders = {}

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        for name, code in self.folders.items():
            os.makedirs(os.path.join(self.tmp, name))
            with open(os.path.join(self.tmp, name, "backend.py"), "w") as f:
                f.write(code)
        patches = [mock.patch.object(backends, "__path__", [self.tmp]), mock.patch.object(backends, "_found", None)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        for name in self.folders:
            self.addCleanup(sys.modules.pop, f"backends.{name}.backend", None)
            self.addCleanup(sys.modules.pop, f"backends.{name}", None)
        self.addCleanup(shutil.rmtree, self.tmp)

    def args(self, *argv):
        with mock.patch.object(sys, "argv", ["install.py", *argv]):
            return install.parser().parse_args()


class NoBackends(WithBackends):
    def test_nothing_found(self):
        self.assertEqual(backends.found(), [])
        self.assertEqual(backends.call("start", {}), [])

    def test_the_installer_offers_what_it_always_did(self):
        backend = next(a for a in install.parser()._actions if a.dest == "backend")
        self.assertEqual(backend.choices, ["lmstudio", "llama", "skip"])
        self.assertEqual(self.args().voice, None)            # and voice() reads None as "install"

    def test_the_launcher_starts_what_it_always_did(self):
        self.assertEqual(launch.start_backends(), [])


class Broken(WithBackends):
    folders = {"oops": "raise RuntimeError('no')\n", "fine": "def backends(inst):\n    return []\n"}

    def test_a_broken_one_is_said_and_left_out(self):
        err = io.StringIO()
        with redirect_stderr(err):
            found = backends.found()
        self.assertEqual([b.__name__ for b in found], ["backends.fine.backend"])
        self.assertIn("backends/oops didn't load", err.getvalue())

    def test_a_start_that_raises_never_holds_her_up(self):
        with open(os.path.join(self.tmp, "fine", "backend.py"), "a") as f:
            f.write("def start(settings):\n    raise OSError('podman went away')\n")
        with redirect_stderr(io.StringIO()):
            self.assertEqual(launch.start_backends(), [])


class Hooks(WithBackends):
    folders = {"pretend": PRETEND}

    def setUp(self):
        super().setUp()
        self.mod = backends.found()[0]
        self.mod.CALLS.clear()

    def run_quietly(self, fn, *args):
        with redirect_stdout(io.StringIO()):
            return fn(*args)

    def test_its_backends_are_choices_and_its_flags_are_there(self):
        a = self.args("--backend", "elsewhere", "--pretend-addr", "10.0.0.2")
        self.assertEqual((a.backend, a.pretend_addr), ("elsewhere", "10.0.0.2"))

    def test_its_errors_stop_the_installer(self):
        with mock.patch.object(sys, "argv", ["install.py", "--backend", "elsewhere"]), \
             redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit):
                install.main()

    def test_the_menu_offers_its_menu_ones_before_skip_with_its_default(self):
        install.ARGS = self.args()
        seen = {}

        def choose(question, options, default):
            seen.update(options=[k for k, _l in options], default=default)
            return default

        state = {}
        with mock.patch.object(install, "choose", choose):
            self.run_quietly(install.backend, state)
        self.assertEqual(seen["options"], ["lmstudio", "llama", "boxes", "skip"])
        self.assertEqual(seen["default"], "boxes")
        self.assertEqual((state["backend"], self.mod.CALLS), ("boxes", [("backend", "boxes")]))

    def test_its_model_gets_the_pick_with_the_gpu_kept_for_it(self):
        install.ARGS = self.args("--yes", "--backend", "boxes")
        state = {"backend": "boxes"}
        with mock.patch.object(install, "detect_vram", return_value=(8.0, "test")):
            self.run_quietly(install.model, state)
        # 8 GB less the 1 GB kept: the 4B, not the 9B Q4 that 8 GB alone would get
        self.assertEqual(self.mod.CALLS, [("model", "4b")])

    def test_a_model_that_lives_elsewhere_skips_the_pick(self):
        install.ARGS = self.args("--yes", "--backend", "elsewhere", "--pretend-addr", "x")
        with mock.patch.object(install, "detect_vram") as vram:
            self.run_quietly(install.model, {"backend": "elsewhere"})
        vram.assert_not_called()
        self.assertEqual(self.mod.CALLS, [("model", None)])

    def test_its_voice_is_offered_first_and_takes_over(self):
        install.ARGS = self.args("--yes")
        state = {}
        with mock.patch.object(install, "http_ok", return_value=False):
            self.run_quietly(install.voice, state, "cpu")
        self.assertEqual(self.mod.CALLS, [("voice", "boxed")])
        self.assertEqual(state["voice"]["mode"], "external")

    def test_a_voice_flag_of_main_s_is_main_s(self):
        install.ARGS = self.args("--yes", "--voice", "skip")
        state = {}
        self.run_quietly(install.voice, state, "cpu")
        self.assertEqual((state["voice"], self.mod.CALLS), ({"mode": "none"}, []))

    def test_the_launcher_leaves_what_it_handled_alone(self):
        started = launch.start_backends()
        self.assertEqual([b["handled"] for b in started], [{"chat"}])
        started[0]["stop"]()
        self.assertEqual(self.mod.CALLS, [("start",), ("stop",)])


class InstallSh(unittest.TestCase):
    """install.sh, in a copy of its folder with a python3 too new for pick():
    a backends/<name>/python.sh is what decides the Python, or nothing does."""

    def sh(self, hook=None):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        os.makedirs(os.path.join(tmp, "installer"))
        shutil.copy(os.path.join(ROOT, "installer", "install.sh"), os.path.join(tmp, "installer"))
        bin_ = os.path.join(tmp, "bin")
        os.makedirs(bin_)
        for name, body in (("python3", "exit 1"), ("newpy", 'echo "ran $*"')):    # python3: "3.14"
            with open(os.path.join(bin_, name), "w") as f:
                f.write(f"#!/bin/sh\n{body}\n")
            os.chmod(os.path.join(bin_, name), 0o755)
        if hook:
            os.makedirs(os.path.join(tmp, "backends", "new"))
            with open(os.path.join(tmp, "backends", "new", "python.sh"), "w") as f:
                f.write(hook)
        os.symlink(shutil.which("dirname"), os.path.join(bin_, "dirname"))
        env = {"PATH": bin_, "HOME": tmp}         # no real Python, nor dnf/apt to offer one
        return subprocess.run(["/bin/bash", os.path.join(tmp, "installer", "install.sh"), "--yes"], env=env,
                              input="n\n", capture_output=True, text=True, timeout=30)

    def test_a_hook_can_pick_the_python(self):
        out = self.sh('PY=newpy\n')
        self.assertEqual((out.returncode, out.stdout.strip()), (0, "ran installer/install.py --yes"))

    def test_without_one_it_still_asks_for_3_12(self):
        out = self.sh()
        self.assertNotEqual(out.returncode, 0)
        self.assertIn("Luna needs Python 3.10-3.13", out.stdout)


if __name__ == "__main__":
    unittest.main()
