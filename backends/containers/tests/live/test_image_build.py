"""The images build, carry their labels, and take only what they need.

plans/container-migration-plan-v2.md, 5.3.
"""
import os
import shutil
import sys
import tempfile
import unittest

from backends.containers.tests.live import support
from backends.containers.tests.live.support import live, podman

LLAMA_DIR = os.path.join(support.ROOT, "backends", "containers", "llama")


@live
class LlamaImage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.image = support.build_llama()

    def test_rootless(self):
        rootless = podman("info", "--format", "{{.Host.Security.Rootless}}").stdout.strip()
        self.assertEqual(rootless, "true")

    def test_labels_name_it_ours(self):
        labels = podman("image", "inspect", "-f", "{{json .Labels}}", self.image).stdout
        self.assertIn('"luna.image-version":"test"', labels)
        self.assertIn('"luna.base-digest":"sha256:', labels)
        self.assertIn('"luna.backend":"server"', labels)

    def test_entrypoint_by_absolute_path(self):
        entry = podman("image", "inspect", "-f", "{{json .Config.Entrypoint}}", self.image).stdout
        self.assertEqual(entry.strip(), '["/luna/entrypoint.sh"]')
        with open(os.path.join(LLAMA_DIR, "entrypoint.sh")) as f:
            self.assertIn("BIN=/app/llama-server", f.read())

    def test_no_user(self):
        user = podman("image", "inspect", "-f", "{{.Config.User}}", self.image).stdout.strip()
        self.assertIn(user, ("", "root", "0"))

    def test_context_keeps_out_everything_but_the_entrypoint(self):
        # The real .containerignore, over a context salted with what must
        # never get in; a probe image copies the whole context to look.
        with tempfile.TemporaryDirectory() as ctx:
            for name in (".containerignore", "entrypoint.sh"):
                shutil.copy(os.path.join(LLAMA_DIR, name), ctx)
            for decoy in ("config.json", ".api_key", "hf_token", "model.gguf"):
                with open(os.path.join(ctx, decoy), "w") as f:
                    f.write("secret")
            base = podman("image", "inspect", "-f", "{{.Id}}", self.image).stdout.strip()
            with open(os.path.join(ctx, "Containerfile.probe"), "w") as f:
                f.write(f"FROM {base}\nCOPY . /ctx\n")
            podman("build", "-q", "-f", os.path.join(ctx, "Containerfile.probe"),
                   "-t", "localhost/luna-test-probe", ctx)
            try:
                found = podman("run", "--rm", "--entrypoint", "ls", "localhost/luna-test-probe",
                               "-A", "/ctx").stdout.split()
            finally:
                podman("rmi", "-f", "localhost/luna-test-probe", check=False)
        self.assertEqual(found, ["entrypoint.sh"])

    def test_entrypoint_is_shellcheck_clean(self):
        out = podman("run", "--rm", "--security-opt", "label=disable",
                     "-v", f"{LLAMA_DIR}:/mnt:ro", "docker.io/koalaman/shellcheck:stable",
                     "/mnt/entrypoint.sh", check=False)
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)

    def test_curl_is_there_for_healthchecks(self):
        out = podman("run", "--rm", "--entrypoint", "sh", self.image, "-c", "command -v curl")
        self.assertEqual(out.stdout.strip(), "/usr/bin/curl")


@live
class TtsImage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.image = support.build_tts()

    def run_in(self, script):
        return podman("run", "--rm", "--network", "none", "--entrypoint", "sh", self.image, "-c", script).stdout

    def test_every_pin_survived(self):
        out = self.run_in("pip show kokoro misaki numpy torch | grep -E '^(Name|Version)'")
        for pin in ("kokoro\nVersion: 0.9.4", "misaki\nVersion: 0.9.4", "numpy\nVersion: 2.5.4",
                    "torch\nVersion: 2.14.1+cpu"):
            self.assertIn(pin.replace("\n", "\n"), out.replace("Name: ", ""))

    def test_cpu_image_has_no_cuda(self):
        out = self.run_in("pip list 2>/dev/null")
        self.assertNotRegex(out.lower(), r"nvidia|cuda")

    def test_kokoro_reader_at_its_pinned_commit_without_git(self):
        out = self.run_in("ls /app/server; command -v git || echo no-git; ls -a /app")
        self.assertIn("kokoro_server.py", out)
        self.assertIn("blends.json", out)
        self.assertIn("no-git", out)
        self.assertNotIn(".git\n", out)
        with open(os.path.join(support.ROOT, "backends", "containers", "tts", "Dockerfile")) as f:
            self.assertRegex(f.read(), r"ARG KOKORO_REF=[0-9a-f]{40}")

    def test_spacy_model_is_in_the_image(self):
        self.assertIn("en_core_web_sm", self.run_in("pip list 2>/dev/null"))

    def test_context_is_empty(self):
        with open(os.path.join(support.ROOT, "backends", "containers", "tts", ".containerignore")) as f:
            self.assertIn("*", f.read().split())

    def test_size_is_recorded(self):
        size = int(podman("image", "inspect", "-f", "{{.Size}}", self.image).stdout)
        print(f"\n  voice image (cpu): {size / 1e9:.2f} GB", file=sys.stderr)
        self.assertLess(size, 3e9)          # the old nvidia/cuda-based one was 11.4 GB


if __name__ == "__main__":
    unittest.main()
