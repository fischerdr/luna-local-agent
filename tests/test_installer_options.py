"""The installer's flags against each other (installer/install.py,
validate_args): main's flags keep their meaning, the new ones add to
them, and the old container branch's breakages stay fixed. Nothing is
installed; Podman, the GPU and the network are pretended.

    python -m unittest discover tests        (from the repo folder)
"""
import contextlib
import importlib.util
import io
import itertools
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.offline import FAKE  # noqa: E402,F401  (first: no real servers)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


install = load(os.path.join(ROOT, "installer", "install.py"), "luna_installer_options")


def check(*argv, system="Linux", nvidia=True, kfd=False):
    args = install.parser().parse_args(list(argv))
    return install.validate_args(args, system=system,
                                 which=lambda name: "/usr/bin/" + name if nvidia and name == "nvidia-smi" else None,
                                 exists=lambda path: kfd and path == "/dev/kfd")


class MainsFlags(unittest.TestCase):
    def test_no_flags_is_mains_defaults(self):
        args, errors, notes = check()
        self.assertEqual((args.backend, args.voice, args.gpu, args.extras, args.model), (None, None, "auto", None, "auto"))
        self.assertEqual((errors, notes), ([], []))

    def test_yes_changes_no_default(self):
        args, errors, notes = check("--yes")
        self.assertEqual((args.backend, args.voice, args.extras), (None, None, None))
        self.assertEqual((errors, notes), ([], []))

    def test_voice_url_gets_through(self):
        args, errors, _ = check("--voice", "url:http://192.168.1.20:8899")
        self.assertEqual(errors, [])
        self.assertEqual(args.voice, "url:http://192.168.1.20:8899")

    def test_voice_typo_is_refused_not_installed(self):
        for typo in ("instal", "contianer", "url:", "url:ftp://x", "http://x:8899"):
            with self.subTest(voice=typo):
                _, errors, _ = check("--voice", typo)
                self.assertEqual(len(errors), 1)

    def test_backend_llama_still_means_the_host_build(self):
        args, errors, _ = check("--backend", "llama")
        self.assertEqual((args.backend, errors), ("llama", []))
        install.ARGS = args
        with mock.patch.object(install, "run", return_value=True) as run, \
             mock.patch.object(install, "SYSTEM", "Linux"), \
             mock.patch.object(install, "detect_gpu", return_value=("cuda", "NVIDIA")), \
             contextlib.redirect_stdout(io.StringIO()):
            install.backend({})
        self.assertEqual(run.call_args[0][0][:2], ["bash", os.path.join(ROOT, "llama", "install.sh")])

    def test_there_is_no_llama_flag(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            install.parser().parse_args(["--llama"])

    def test_last_repeated_flag_wins(self):
        args, _, _ = check("--backend", "llama", "--backend", "lmstudio")
        self.assertEqual(args.backend, "lmstudio")


class Remote(unittest.TestCase):
    def test_remote_alone_implies_the_backend_and_the_voice(self):
        args, errors, _ = check("--remote", "192.168.1.50")
        self.assertEqual(errors, [])
        self.assertEqual(args.backend, "remote")
        self.assertEqual(args.voice, "url:http://192.168.1.50:8899")

    def test_ipv6_and_hostnames(self):
        self.assertEqual(check("--remote", "fd00::5")[0].voice, "url:http://[fd00::5]:8899")
        self.assertEqual(check("--remote", "gpu-box.lan")[0].voice, "url:http://gpu-box.lan:8899")

    def test_an_explicit_voice_wins(self):
        args, _, _ = check("--remote", "192.168.1.50", "--voice", "skip")
        self.assertEqual(args.voice, "skip")

    def test_remote_with_another_backend_is_an_error(self):
        _, errors, _ = check("--remote", "192.168.1.50", "--backend", "llama")
        self.assertEqual(len(errors), 1)

    def test_backend_remote_without_an_address(self):
        self.assertEqual(len(check("--backend", "remote", "--yes")[1]), 1)
        self.assertEqual(check("--backend", "remote")[1], [])      # asked for interactively

    def test_this_machine_is_not_remote(self):
        for addr in ("127.0.0.1", "localhost", "::1", "0.0.0.0", "bad host!"):
            with self.subTest(addr=addr):
                self.assertEqual(len(check("--remote", addr)[1]), 1)


class Containers(unittest.TestCase):
    def test_linux_only(self):
        for system in ("Darwin", "Windows"):
            with self.subTest(system=system):
                errors = check("--backend", "container", system=system)[1]
                self.assertEqual(len(errors), 1)
                self.assertIn("--remote", errors[0])
                self.assertEqual(len(check("--voice", "container", system=system)[1]), 1)

    def test_no_podman_is_not_an_error(self):
        with mock.patch.object(install.shutil, "which", return_value=None):
            self.assertEqual(check("--backend", "container")[1], [])

    def test_mps_with_containers_is_an_error(self):
        self.assertEqual(len(check("--backend", "container", "--gpu", "mps")[1]), 1)
        self.assertEqual(check("--gpu", "mps")[1], [])          # the host voice: as on main

    def test_gpu_that_isnt_there_is_a_note(self):
        _, errors, notes = check("--backend", "container", "--gpu", "cuda", nvidia=False)
        self.assertEqual(errors, [])
        self.assertIn("cpu image", notes[0])
        _, _, notes = check("--backend", "container", "--gpu", "rocm", kfd=False)
        self.assertIn("/dev/kfd", notes[0])
        self.assertEqual(check("--backend", "container", "--gpu", "rocm", kfd=True)[2], [])

    def test_gpu_flag_reaches_the_containers(self):
        for flag, want in (("cpu", "cpu"), ("auto", "auto"), ("cuda", "auto")):
            with self.subTest(gpu=flag):
                install.ARGS = check("--backend", "container", "--gpu", flag)[0]
                with mock.patch.object(install.shutil, "which", return_value="/usr/bin/nvidia-smi"):
                    self.assertEqual(install.container_gpu(), want)

    def test_no_podman_doesnt_crash(self):
        install.ARGS = check("--backend", "container")[0]
        install.REPORT.clear()
        with mock.patch.object(install.shutil, "which", return_value=None), \
             contextlib.redirect_stdout(io.StringIO()) as out:
            install.containers({})
            self.assertFalse(install.voice_container({}))
        self.assertIn("Podman isn't installed", out.getvalue())
        self.assertFalse(install.REPORT[0][1])


class Extras(unittest.TestCase):
    def test_containers_and_remote_leave_out_the_side_models(self):
        for backend in ("container", "remote"):
            with self.subTest(backend=backend):
                more = ["--remote", "10.0.0.2"] if backend == "remote" else []
                args, _, notes = check("--backend", backend, *more, "--extras", "embeddings,mcp,reranker")
                self.assertEqual(args.extras, "mcp")
                self.assertIn("embeddings, reranker", notes[0])

    def test_lmstudio_and_llama_keep_them(self):
        for backend in ("lmstudio", "llama"):
            args, _, notes = check("--backend", backend, "--extras", "embeddings,reranker")
            self.assertEqual((args.extras, notes), ("embeddings,reranker", []))

    def test_unknown_extras_are_named(self):
        args, _, notes = check("--extras", "mcp,wakewrod")
        self.assertEqual(args.extras, "mcp")
        self.assertIn("wakewrod", notes[0])

    def test_all(self):
        args, _, _ = check("--backend", "container", "--extras", "all")
        self.assertNotIn("embeddings", args.extras)
        self.assertIn("mcp", args.extras)


class CheckOnly(unittest.TestCase):
    def test_check_with_other_flags_names_them(self):
        args, errors, notes = check("--check", "--backend", "container", "--voice", "bogus")
        self.assertEqual(errors, [])
        self.assertIn("--backend", notes[0])
        self.assertIn("--voice", notes[0])

    def test_check_alone(self):
        self.assertEqual(check("--check")[1:], ([], []))


class Matrix(unittest.TestCase):
    """Every flag against every other: no surprises, ever."""

    def test_everything(self):
        voices = (None, "install", "container", "url:http://h:1", "skip", "bogus")
        for backend, voice, gpu, remote, yes, system in itertools.product(
                (None,) + install.BACKENDS, voices, ("auto", "cpu", "cuda", "rocm", "mps"),
                (None, "10.0.0.2", "127.0.0.1"), (False, True), ("Linux", "Darwin", "Windows")):
            argv = [*(["--backend", backend] if backend else []), *(["--voice", voice] if voice else []),
                    "--gpu", gpu, *(["--remote", remote] if remote else []), *(["--yes"] if yes else [])]
            args, errors, _ = check(*argv, system=system)
            with self.subTest(argv=argv, system=system):
                if voice == "bogus":
                    self.assertTrue(errors)
                if remote and backend not in (None, "remote"):
                    self.assertTrue(errors)
                if not errors:
                    if remote:
                        self.assertEqual(args.backend, "remote")
                    if args.backend == "container" or args.voice == "container":
                        self.assertEqual(system, "Linux")
                        self.assertNotEqual(gpu, "mps")
                    if backend in ("lmstudio", "llama", "skip"):
                        self.assertEqual(args.backend, backend)       # main's meaning, kept
                    if voice not in (None,):
                        self.assertEqual(args.voice, voice)            # an explicit --voice always wins


class SystemPackages(unittest.TestCase):
    def test_apt_gets_this_pythons_dev_package(self):
        install.ARGS = check("--no-system")[0]
        mm = f"{sys.version_info.major}.{sys.version_info.minor}"
        seen = []
        with mock.patch.object(install, "package_manager", return_value=("apt-get", ["apt-get", "install"])), \
             mock.patch.object(install, "installed", side_effect=lambda pm, p: seen.append(p) or True), \
             mock.patch.object(install, "SYSTEM", "Linux"), \
             mock.patch.object(install.shutil, "which", return_value="/usr/bin/x"), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(install.system_packages())
        self.assertIn(f"python{mm}-dev", seen)
        self.assertIn(f"python{mm}-venv", seen)
        self.assertNotIn("python3-dev", seen)


class CheckOutputUnchanged(unittest.TestCase):
    """install.py --check on a main-style install says what main's did."""

    def setUp(self):
        base = subprocess.run(["git", "-C", ROOT, "merge-base", "HEAD", "main"], capture_output=True, text=True)
        rev = os.environ.get("FENCE_BASE") or base.stdout.strip()
        src = subprocess.run(["git", "-C", ROOT, "show", f"{rev}:installer/install.py"],
                             capture_output=True, text=True) if rev else None
        if not src or src.returncode:
            self.skipTest("main's installer isn't in this checkout's history")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        os.makedirs(os.path.join(self.tmp.name, "installer"))
        path = os.path.join(self.tmp.name, "installer", "install.py")
        with open(path, "w") as f:
            f.write(src.stdout)
        self.main = load(path, "luna_installer_main")

    def health(self, module, state):
        module.ARGS = module.argparse.Namespace(yes=True)
        done = subprocess.CompletedProcess([], 0, '{}\n', "")
        with mock.patch.object(module, "http_ok", side_effect=lambda url, timeout=2: "8899" in url), \
             mock.patch.object(module.subprocess, "run", return_value=done), \
             contextlib.redirect_stdout(io.StringIO()) as out:
            module.health(state)
        return out.getvalue()

    def test_same_lines(self):
        for state in ({"backend": "llama", "venv_python": "/x/python",
                       "voice": {"mode": "managed", "python": "/k/python", "server": "/k/s.py"}},
                      {"backend": "lmstudio", "venv_python": "/x/python",
                       "voice": {"mode": "external", "url": "http://127.0.0.1:8899"}}):
            with self.subTest(state=state["backend"]):
                self.assertEqual(self.health(install, state), self.health(self.main, state))


if __name__ == "__main__":
    unittest.main()
