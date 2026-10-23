"""installer/install.sh picking a Python, with this backend's python.sh:
3.10-3.13 as on main; with only a newer one, Podman means going on with
it (the containers carry what needs the older one) instead of asking for
sudo; without Podman, main's offer. Run on a PATH of pretend pythons and
podman, so it's the same here as on any test machine.

    python -m unittest discover -s backends/containers/tests -t .
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from tests.offline import FAKE  # noqa: E402,F401  (first: no real servers)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# A pretend Python of a given version: answers install.sh's version checks
# by running them on a real Python with that version_info, and stands in
# for install.py by saying how it was called.
PRETEND = """#!/bin/bash
if [ "$1" = installer/install.py ]; then echo "RAN install.py with $(basename "$0") $*"; exit 0; fi
if [ "$1" = -c ]; then
    exec {real} -c "import sys, collections
V = collections.namedtuple('V', 'major minor micro releaselevel serial')
sys.version_info = V({major}, {minor}, 0, 'final', 0)
exec(sys.argv[1])" "$2"
fi
"""


class InstallSh(unittest.TestCase):
    def run_sh(self, pythons, podman, answer="n"):
        with tempfile.TemporaryDirectory() as tmp:
            repo = os.path.join(tmp, "repo")
            os.makedirs(os.path.join(repo, "installer"))
            shutil.copy(os.path.join(ROOT, "installer", "install.sh"), os.path.join(repo, "installer"))
            os.makedirs(os.path.join(repo, "backends", "containers"))
            shutil.copy(os.path.join(ROOT, "backends", "containers", "python.sh"),
                        os.path.join(repo, "backends", "containers"))
            bin_dir = os.path.join(tmp, "bin")
            os.makedirs(bin_dir)
            for name, (major, minor) in pythons.items():
                path = os.path.join(bin_dir, name)
                with open(path, "w") as f:
                    f.write(PRETEND.format(real=sys.executable, major=major, minor=minor))
                os.chmod(path, 0o755)
            for tool in ("dirname", "basename") + (("podman",) if podman else ()):
                os.symlink(shutil.which(tool) or "/bin/true", os.path.join(bin_dir, tool))
            # pretend dnf, so main's offer names its command
            os.symlink("/bin/true", os.path.join(bin_dir, "dnf"))
            r = subprocess.run(["/bin/bash", os.path.join(repo, "installer", "install.sh"), "--yes"],
                               cwd=repo, env={"PATH": bin_dir, "HOME": tmp}, input=answer + "\n",
                               capture_output=True, text=True, timeout=30)
        return r.returncode, r.stdout + r.stderr

    def test_a_supported_python_is_used_as_on_main(self):
        code, out = self.run_sh({"python3.12": (3, 12), "python3": (3, 14)}, podman=True)
        self.assertEqual(code, 0)
        self.assertIn("RAN install.py with python3.12 installer/install.py --yes", out)
        self.assertNotIn("containers", out)

    def test_only_a_newer_python_and_podman_goes_on_with_it(self):
        code, out = self.run_sh({"python3": (3, 14)}, podman=True)
        self.assertEqual(code, 0, out)
        self.assertIn("RAN install.py with python3 installer/install.py --yes", out)
        self.assertIn("Podman containers", out)
        self.assertNotIn("sudo", out)

    def test_only_a_newer_python_without_podman_is_mains_offer(self):
        code, out = self.run_sh({"python3": (3, 14)}, podman=False, answer="n")
        self.assertEqual(code, 1)                      # main's offer, declined
        self.assertIn("none was found", out)
        self.assertNotIn("RAN install.py", out)

    def test_an_older_python_is_never_enough(self):
        code, out = self.run_sh({"python3": (3, 9)}, podman=True, answer="n")
        self.assertEqual(code, 1)
        self.assertNotIn("RAN install.py", out)


if __name__ == "__main__":
    unittest.main()
