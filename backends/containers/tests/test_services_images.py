"""Which image this machine gets (images.py): the five steps, from
a pretend GPU, CDI spec and /dev - nothing here looks at the real ones.

    python -m unittest discover tests        (from the repo folder)
"""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from tests.offline import FAKE  # noqa: E402,F401  (first: no real servers)

from backends.containers import images  # noqa: E402


def pick(override="auto", cuda=None, cdi=None, devs=(), arch="x86_64", gfx=("gfx1030",)):
    return images.image_for_host(override, cuda=lambda: cuda, cdi=lambda: cdi,
                                 exists=lambda p: p in devs, arch=lambda: arch, gfx=lambda: list(gfx))


class Detection(unittest.TestCase):
    def test_nvidia_cuda13(self):
        self.assertEqual(pick(cuda=13, cdi="/etc/cdi/nvidia.yaml"), ("cuda13", None))

    def test_nvidia_cuda12(self):
        self.assertEqual(pick(cuda=12, cdi="/etc/cdi/nvidia.yaml"), ("cuda", None))

    def test_nvidia_without_cdi_is_cpu_with_the_command(self):
        backend, note = pick(cuda=13, devs=("/dev/dri",))
        self.assertEqual(backend, "cpu")          # never Vulkan on NVIDIA
        self.assertIn("nvidia-ctk cdi generate", note)

    def test_rocm(self):
        self.assertEqual(pick(devs=("/dev/kfd", "/dev/dri")), ("rocm", None))

    def test_arm64_has_no_rocm_image(self):
        self.assertEqual(pick(devs=("/dev/kfd", "/dev/dri"), arch="aarch64"), ("vulkan", None))

    def test_vulkan(self):
        self.assertEqual(pick(devs=("/dev/dri",)), ("vulkan", None))

    def test_nothing_is_cpu(self):
        self.assertEqual(pick(), ("cpu", None))


class Rocm(unittest.TestCase):
    def test_a_gpu_the_image_has(self):
        self.assertEqual(pick(devs=("/dev/kfd",), gfx=("gfx1100",)), ("rocm", None))

    def test_a_gpu_it_wasnt_built_for_says_so(self):
        backend, note = pick(devs=("/dev/kfd",), gfx=("gfx1031",))
        self.assertEqual(backend, "rocm")
        self.assertIn("gfx1031", note)
        self.assertIn("vulkan", note)
        self.assertIsNotNone(pick("rocm", devs=("/dev/kfd",), gfx=("gfx803",))[1])

    def test_reading_rocminfo(self):
        out = "Agent 1\n  Name:                    AMD Ryzen\nAgent 2\n  Name:                    gfx1030\n"
        with mock.patch.object(images.shutil, "which", return_value="/usr/bin/rocminfo"), \
             mock.patch.object(images.subprocess, "run", return_value=mock.Mock(stdout=out)):
            self.assertEqual(images.rocm_gpus(), ["gfx1030"])


class Override(unittest.TestCase):
    def test_override_wins(self):
        self.assertEqual(pick("cpu", cuda=13, cdi="/etc/cdi/nvidia.yaml"), ("cpu", None))
        self.assertEqual(pick("vulkan", devs=("/dev/dri",)), ("vulkan", None))

    def test_override_that_cant_work_says_so(self):
        backend, note = pick("cuda")
        self.assertEqual(backend, "cpu")
        self.assertIn("no NVIDIA", note)
        backend, note = pick("rocm")
        self.assertEqual(backend, "cpu")
        self.assertIn("/dev/kfd", note)
        backend, note = pick("mps")
        self.assertEqual(backend, "cpu")
        self.assertIn("mps", note)


class Build(unittest.TestCase):
    def test_every_base_pinned_by_digest(self):
        for backend, (tag, digest) in images.BASES.items():
            with self.subTest(backend=backend):
                self.assertTrue(tag.startswith("server"))
                self.assertRegex(digest, r"^sha256:[0-9a-f]{64}$")

    def test_build_labels_and_context(self):
        cmd = images.build_cmd("cuda13")
        self.assertIn("BASE_DIGEST=" + images.BASES["cuda13"][1], cmd)
        self.assertIn("luna.backend=cuda13", cmd)
        self.assertIn("luna.image-version=" + images.image_version(), cmd)
        self.assertEqual(cmd[-1], images.LLAMA_CONTEXT)      # never the repo root

    def test_our_tag_never_moves_an_older_setups(self):
        # The old branch's retag was localhost/luna-llama:cuda13.
        for backend in images.BASES:
            self.assertNotIn("/luna-llama:", images.tag(backend))

    def test_version_follows_the_build_files(self):
        self.assertRegex(images.image_version(), r"^[0-9a-f]{16}$")
        self.assertEqual(images.image_version(), images.image_version())


if __name__ == "__main__":
    unittest.main()
