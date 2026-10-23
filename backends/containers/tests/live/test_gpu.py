"""The GPU image on a real GPU: it starts, its layers go to the GPU, and
bench makes a recommendation. Skipped without an NVIDIA CDI spec or
/dev/kfd. A tiny model, so it fits beside whatever else is loaded.
"""
import os
import subprocess
import unittest

from backends.containers import bench, images, run
from backends.containers.tests.live import support
from backends.containers.tests.live.support import live, podman

BACKEND, _ = images.image_for_host("auto")
GPU = BACKEND != "cpu" and (images.cdi_spec() or os.path.exists("/dev/kfd"))


@live
@unittest.skipUnless(GPU, "no NVIDIA CDI spec or /dev/kfd here")
class Gpu(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mem = bench.vram()
        if mem and mem[0] - mem[1] < 1200:
            raise unittest.SkipTest(f"the GPU is busy: {mem[1]} of {mem[0]} MiB in use")
        cls.image = images.find(BACKEND)
        if not cls.image:
            subprocess.run(images.build_cmd(BACKEND), check=True, capture_output=True)
            cls.image = images.find(BACKEND)
        support.make_key()
        selinux = podman("info", "--format", "{{.Host.Security.SELinuxEnabled}}").stdout.strip() == "true"
        flags = run.gpu_flags(BACKEND, selinux, render=False)
        try:
            podman("run", "-d", "--replace", "--name", "luna-test-chat",
                   "-p", f"127.0.0.1:{support.PORTS['chat']}:8080", "-v", f"{support.MODELS}:/models",
                   "--secret", f"{support.KEY_SECRET},target=api_key", *flags,
                   "-e", "ROLE=chat", "-e", f"LUNA_HF={support.models()['TEST_CHAT_HF']}",
                   "-e", "CTX=2048", "-e", "PARALLEL=1", "-e", "NGL=99", cls.image)
            support.wait_ready("chat")
        except Exception:
            support.cleanup("chat")
            raise

    @classmethod
    def tearDownClass(cls):
        support.cleanup("chat")

    @unittest.skipUnless(BACKEND in images.NVIDIA, "the VRAM check here reads nvidia-smi")
    def test_the_model_is_in_vram(self):
        # CDI puts nvidia-smi in the container; llama-server shows as using the GPU.
        out = podman("exec", "luna-test-chat", "nvidia-smi", "--query-compute-apps=used_memory",
                     "--format=csv,noheader,nounits").stdout
        used = [int(x) for x in out.split() if x.isdigit()]
        self.assertTrue(used and max(used) > 100, f"nothing on the GPU: {out!r}")

    def test_bench_recommends(self):
        mem = bench.vram()
        self.assertIsNotNone(mem)
        total, used = mem
        place, why = bench.recommend(total, used, 1000, 0.8)
        self.assertEqual(place["EMBED_NGL"], 0)
        self.assertIn(place["RERANK_NGL"], (0, 99))
        self.assertTrue(why)


if __name__ == "__main__":
    unittest.main()
