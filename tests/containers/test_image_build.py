"""The images build, carry their labels, and take only what they need.

plans/container-migration-plan-v2.md, 5.3.
"""
import os
import shutil
import tempfile
import unittest

from tests.containers import support
from tests.containers.support import live, podman

LLAMA_DIR = os.path.join(support.ROOT, "services", "llama")


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


if __name__ == "__main__":
    unittest.main()
