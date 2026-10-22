"""The podman run for each role (services/run.py): published on HOST only,
secrets and volumes, the GPU flags, and a config hash that changes with
everything that matters - and never a key or token in sight.

    python -m unittest discover tests        (from the repo folder)
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.offline import FAKE  # noqa: E402,F401  (first: no real servers)

from services import run, settings  # noqa: E402

KEY = "k3y-that-must-never-show"
TOKEN = "hf_token-that-must-never-show"


def base_settings(**more):
    s = {k: "" for k in settings.KNOWN}
    s.update(settings.DEFAULTS)
    s.update({"NGL": "99", "EMBED_NGL": "0", "RERANK_NGL": "99", "THREADS": "9"})
    s.update(more)
    return s


def argv(role="chat", s=None, model=None, **kw):
    s = s or base_settings()
    model = model or settings.Model(hf="org/repo:Q4")
    args = dict(image="IMG", image_id="sha256:abc", backend="cpu", env_file="/run/user/1/luna/x.env",
                env_text=f"ROLE={role}\n", key_fp="kfp", token_fp="")
    args.update(kw)
    return run.run_args(role, s, model, **args)


class Publish(unittest.TestCase):
    def test_loopback_by_default(self):
        a = argv()
        self.assertEqual(a[a.index("-p") + 1], "127.0.0.1:8080:8080")

    def test_each_role_on_its_port(self):
        for role, port in (("chat", 8080), ("embed", 8081), ("rerank", 8082)):
            with self.subTest(role=role):
                a = argv(role)
                self.assertEqual(a[a.index("-p") + 1], f"127.0.0.1:{port}:8080")

    def test_host_and_ports_from_server_env(self):
        a = argv("embed", base_settings(HOST="192.168.1.5", EMBED_PORT="9001"))
        self.assertEqual(a[a.index("-p") + 1], "192.168.1.5:9001:8080")

    def test_ipv6_in_brackets(self):
        a = argv(s=base_settings(HOST="::1"))
        self.assertEqual(a[a.index("-p") + 1], "[::1]:8080:8080")

    def test_never_host_network(self):
        for role in ("chat", "embed", "rerank"):
            for backend in ("cpu", "cuda13", "rocm", "vulkan"):
                a = " ".join(argv(role, backend=backend))
                self.assertNotIn("--network", a)
                self.assertNotIn("--net=", a)


class SecretsAndVolumes(unittest.TestCase):
    def test_key_secret_always_token_only_when_there(self):
        a = argv()
        self.assertIn("luna_api_key,target=api_key", a)
        self.assertNotIn("luna_hf_token,target=hf_token", a)
        self.assertIn("luna_hf_token,target=hf_token", argv(token_fp="tfp"))

    def test_volumes(self):
        a = argv()
        self.assertIn("luna-models:/models", a)
        self.assertIn("luna-data:/data", a)
        self.assertNotIn("luna-data:/data", argv("embed"))

    def test_models_dir_read_only_never_relabelled(self):
        s = base_settings(LUNA_MODELS_DIR="/srv/models")
        m = settings.Model(file="/models-host/a/b.gguf", host_mount=True)
        a = argv(s=s, model=m, selinux=True)
        mount = [x for x in a if x.endswith(":/models-host:ro")]
        self.assertEqual(mount, [os.path.realpath("/srv/models") + ":/models-host:ro"])
        self.assertNotIn(":z", " ".join(a))
        self.assertNotIn(":Z", " ".join(a))
        self.assertIn("label=disable", a)
        self.assertNotIn("label=disable", argv(s=s, model=m, selinux=False))

    def test_no_key_or_token_in_argv_or_env(self):
        a = argv(token_fp=run.config_hash([], "", "", "", ""), env_text="ROLE=chat\nCTX=4096\n")
        joined = " ".join(a)
        self.assertNotIn(KEY, joined)
        self.assertNotIn(TOKEN, joined)
        self.assertNotIn("HF_TOKEN", joined)
        self.assertNotIn("api_key=", joined.replace("target=api_key", ""))

    def test_env_file_never_in_the_repo(self):
        a = argv()
        self.assertFalse(a[a.index("--env-file") + 1].startswith(settings.LLAMA_DIR))


class Gpu(unittest.TestCase):
    def test_nvidia_by_cdi(self):
        a = argv(backend="cuda13", selinux=True)
        self.assertIn("nvidia.com/gpu=all", a)
        self.assertIn("label=disable", a)
        self.assertNotIn("--gpus", a)
        self.assertNotIn("--runtime", " ".join(a))

    def test_user_level_cdi_spec(self):
        a = argv(backend="cuda", cdi_dir="/home/me/.config/cdi")
        self.assertEqual(a[:3], ["podman", "--cdi-spec-dir", "/home/me/.config/cdi"])
        self.assertEqual(argv(backend="cpu", cdi_dir="/home/me/.config/cdi")[:2], ["podman", "run"])

    def test_rocm(self):
        a = " ".join(argv(backend="rocm", render=True))
        for flag in ("--device /dev/kfd", "--device /dev/dri", "--group-add video",
                     "seccomp=unconfined", "--group-add render"):
            self.assertIn(flag, a)

    def test_vulkan(self):
        a = " ".join(argv(backend="vulkan"))
        self.assertIn("--device /dev/dri --group-add video", a)
        self.assertNotIn("render", a)
        self.assertNotIn("label=disable", a)      # only on SELinux hosts

    def test_cpu_has_no_devices(self):
        self.assertNotIn("--device", argv(backend="cpu"))

    def test_ngl_zero_stays_off_the_gpu(self):
        s = base_settings()
        self.assertFalse(run.wants_gpu("embed", s, "cuda13"))      # EMBED_NGL=0
        self.assertTrue(run.wants_gpu("rerank", s, "cuda13"))
        self.assertTrue(run.wants_gpu("chat", s, "cuda13"))
        self.assertFalse(run.wants_gpu("chat", s, "cpu"))


class Hash(unittest.TestCase):
    def label(self, a):
        return a[-2]

    def test_stable(self):
        self.assertEqual(self.label(argv()), self.label(argv()))
        self.assertTrue(self.label(argv()).startswith("luna.config-hash="))
        self.assertEqual(argv()[-1], "IMG")

    def test_changes_with_everything_that_matters(self):
        same = self.label(argv())
        for change in (dict(env_text="ROLE=chat\nCTX=1\n"), dict(image_id="sha256:new"),
                       dict(key_fp="other"), dict(token_fp="tfp"), dict(backend="cuda13"),
                       dict(s=base_settings(PORT="9999"))):
            with self.subTest(change=change):
                self.assertNotEqual(self.label(argv(**change)), same)


if __name__ == "__main__":
    unittest.main()
