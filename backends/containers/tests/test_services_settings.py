"""The containers' settings (services/settings.py): the four env files read
by bash in start.sh's order, and which model file each role gets. Every
file is a temporary one; your own server.env is never read.

    python -m unittest discover tests        (from the repo folder)
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from tests.offline import FAKE  # noqa: E402,F401  (first: no real servers)

from backends.containers import settings  # noqa: E402

SERVER_ENV = '''# like llama/server.env
MODEL=""
MODEL_MATCH="*mymodel*"
MMPROJ=""
MMPROJ_MATCH=""
ALIAS=""
HOST="127.0.0.1"          # localhost only
PORT=8080
CTX=16384
NGL=99
THREADS=9
EMBED=auto
EMBED_MODEL=""
EMBED_MATCH="*qwen3-embedding*"
EMBED_PORT=8081
EMBED_CTX=2048
EMBED_POOLING="last"
RERANK=auto
RERANK_MODEL=""
RERANK_MATCH="*qwen3-reranker*"
RERANK_PORT=8082
RERANK_CTX=2048
'''

SERVICES_ENV = '''MODEL_HF=""
MMPROJ_HF=""
EMBED_HF="Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0"
RERANK_HF="ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF:Q8_0"
LUNA_MODELS_DIR=""
EMBED_NGL=0
RERANK_NGL=99
LLAMA_BACKEND=auto
READY_TIMEOUT=600
'''


class Files(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.llama = os.path.join(self.tmp.name, "llama")
        self.services = os.path.join(self.tmp.name, "services")
        os.makedirs(self.llama)
        os.makedirs(self.services)
        self.write("llama/server.env", SERVER_ENV)
        self.write("services/services.env", SERVICES_ENV)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, rel, text):
        path = os.path.join(self.tmp.name, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(text)
        return path

    def load(self, llama_env=""):
        return settings.load(self.llama, self.services, llama_env=llama_env)


class Layering(Files):
    def test_server_env_alone(self):
        s = self.load()
        self.assertEqual(s["PORT"], "8080")
        self.assertEqual(s["HOST"], "127.0.0.1")
        self.assertEqual(s["EMBED_HF"], "Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0")

    def test_each_later_file_wins(self):
        self.write("llama/server.local.env", "CTX=4096\nNGL=10\n")
        self.write("services/services.local.env", "NGL=20\nEMBED_NGL=5\n")
        s = self.load()
        self.assertEqual(s["CTX"], "4096")          # server.local.env over server.env
        self.assertEqual(s["NGL"], "20")            # services.local.env over both
        self.assertEqual(s["EMBED_NGL"], "5")

    def test_llama_env_replaces_server_env(self):
        self.write("llama/other.env", 'PORT=9090\nHOST="127.0.0.1"\n')
        s = self.load(llama_env="other.env")
        self.assertEqual(s["PORT"], "9090")
        self.assertEqual(s["CTX"], "")              # server.env wasn't read at all

    def test_bash_quoting_and_comments(self):
        self.write("llama/server.local.env",
                   'EXTRA="--control-vector a b.gguf"   # a comment\nALIAS=\'it\'"s"\n')
        s = self.load()
        self.assertEqual(s["EXTRA"], "--control-vector a b.gguf")
        self.assertEqual(s["ALIAS"], "its")

    def test_only_known_keys(self):
        self.write("llama/server.local.env", "SOMETHING_ELSE=1\nPATH=/x\n")
        s = self.load()
        self.assertNotIn("SOMETHING_ELSE", s)
        self.assertNotIn("PATH", s)

    def test_a_broken_file_says_so(self):
        self.write("llama/server.local.env", "if then\n")
        with self.assertRaises(settings.SettingsError):
            self.load()

    def test_old_server_env_gets_start_sh_fallbacks(self):
        self.write("llama/server.env", 'HOST="127.0.0.1"\nPORT=8080\n')
        s = self.load()
        self.assertEqual(s["RERANK_PORT"], "8082")
        self.assertEqual(s["RERANK"], "auto")

    def test_non_loopback_host_warns(self):
        self.assertIsNone(settings.host_warning(self.load()))
        self.write("llama/server.local.env", 'HOST="0.0.0.0"\n')
        self.assertIn("0.0.0.0", settings.host_warning(self.load()))


class ModelFiles(Files):
    """A models folder (LUNA_MODELS_DIR) with one chat model in it."""

    def setUp(self):
        super().setUp()
        self.models = os.path.join(self.tmp.name, "models")
        self.gguf("lmstudio/org/My-Model-GGUF/mymodel-Q4.gguf")

    def gguf(self, rel):
        return self.write(os.path.join("models", rel), "gguf")

    def with_dir(self, **more):
        lines = [f'LUNA_MODELS_DIR="{self.models}"'] + [f'{k}="{v}"' for k, v in more.items()]
        self.write("services/services.local.env", "\n".join(lines) + "\n")
        return self.load()


class Models(ModelFiles):
    def test_hf_comes_first(self):
        s = self.with_dir(MODEL_HF="org/repo:Q4_K_M")
        m, _ = settings.resolve("chat", s)
        self.assertEqual(m.hf, "org/repo:Q4_K_M")
        self.assertFalse(m.host_mount)

    def test_a_file_keeps_its_subfolders(self):
        path = os.path.join(self.models, "lmstudio/org/My-Model-GGUF/mymodel-Q4.gguf")
        s = self.with_dir(MODEL=path)
        m, _ = settings.resolve("chat", s)
        self.assertEqual(m.file, "/models-host/lmstudio/org/My-Model-GGUF/mymodel-Q4.gguf")
        self.assertTrue(m.host_mount)

    def test_a_file_already_in_the_volume(self):
        s = self.with_dir(MODEL="/models/imported/x.gguf")
        m, _ = settings.resolve("chat", s)
        self.assertEqual(m.file, "/models/imported/x.gguf")
        self.assertFalse(m.host_mount)

    def test_a_file_outside_the_folder_names_the_folder_to_mount(self):
        other = self.write("elsewhere/x.gguf", "gguf")
        s = self.with_dir(MODEL=other)
        with self.assertRaises(settings.SettingsError) as e:
            settings.resolve("chat", s)
        self.assertIn(os.path.dirname(other), str(e.exception))
        self.assertIn("LUNA_MODELS_DIR", str(e.exception))

    def test_match_finds_one(self):
        m, _ = settings.resolve("chat", self.with_dir())
        self.assertEqual(m.file, "/models-host/lmstudio/org/My-Model-GGUF/mymodel-Q4.gguf")

    def test_match_is_case_insensitive_and_skips_mmproj(self):
        self.gguf("lmstudio/org/My-Model-GGUF/mmproj-MyModel-F16.gguf")
        self.write("services/services.local.env",
                   f'LUNA_MODELS_DIR="{self.models}"\nMODEL_MATCH="*MYMODEL*"\n')
        m, _ = settings.resolve("chat", self.load())
        self.assertTrue(m.file.endswith("mymodel-Q4.gguf"))

    def test_more_than_one_match_is_an_error(self):
        self.gguf("other/mymodel-Q8.gguf")
        with self.assertRaises(settings.SettingsError) as e:
            settings.resolve("chat", self.with_dir())
        self.assertIn("more than one", str(e.exception))

    def test_match_falls_back_to_the_volume(self):
        s = self.with_dir(MODEL_MATCH="*imported*")
        m, _ = settings.resolve("chat", s, volume_find=lambda p: ["/models/x/imported.gguf"])
        self.assertEqual(m.file, "/models/x/imported.gguf")

    def test_no_chat_model_refuses(self):
        s = self.with_dir(MODEL_MATCH="*nothing-like-it*")
        with self.assertRaises(settings.SettingsError) as e:
            settings.resolve("chat", s)
        self.assertIn("MODEL_HF", str(e.exception))

    def test_side_roles_auto_skip_and_on_refuses(self):
        self.write("services/services.local.env",
                   f'LUNA_MODELS_DIR="{self.models}"\nEMBED_HF=""\nRERANK_HF=""\n')
        s = self.load()
        m, why = settings.resolve("embed", s)
        self.assertIsNone(m)
        self.assertIn("qwen3-embedding", why)
        s["RERANK"] = "on"
        with self.assertRaises(settings.SettingsError):
            settings.resolve("rerank", s)

    def test_off_means_off_even_with_a_model(self):
        s = self.with_dir()
        s["EMBED"] = "off"
        m, why = settings.resolve("embed", s)
        self.assertIsNone(m)
        self.assertIn("EMBED=off", why)


class Vision(ModelFiles):
    def test_mmproj_beside_the_model(self):
        self.gguf("lmstudio/org/My-Model-GGUF/mmproj-F16.gguf")
        m, _ = settings.resolve("chat", self.with_dir())
        self.assertEqual(m.mmproj, "/models-host/lmstudio/org/My-Model-GGUF/mmproj-F16.gguf")

    def test_off_turns_vision_off(self):
        self.gguf("lmstudio/org/My-Model-GGUF/mmproj-F16.gguf")
        s = self.with_dir()
        s["MMPROJ"] = "off"
        m, _ = settings.resolve("chat", s)
        self.assertTrue(m.no_mmproj)
        self.assertIn("MMPROJ=off", settings.env_lines("chat", s, m))

    def test_two_beside_the_model_is_an_error(self):
        self.gguf("lmstudio/org/My-Model-GGUF/mmproj-F16.gguf")
        self.gguf("lmstudio/org/My-Model-GGUF/mmproj-Q8.gguf")
        with self.assertRaises(settings.SettingsError):
            settings.resolve("chat", self.with_dir())

    def test_mmproj_by_match(self):
        self.gguf("base/mmproj-Base-F16.gguf")
        s = self.with_dir(MMPROJ_MATCH="*base*")
        m, _ = settings.resolve("chat", s)
        self.assertEqual(m.mmproj, "/models-host/base/mmproj-Base-F16.gguf")

    def test_mmproj_hf(self):
        s = self.with_dir(MMPROJ_HF="org/Base-GGUF/mmproj-F16.gguf")
        m, _ = settings.resolve("chat", s)
        self.assertEqual(m.mmproj_url, "https://huggingface.co/org/Base-GGUF/resolve/main/mmproj-F16.gguf")

    def test_hf_model_brings_its_own(self):
        s = self.with_dir(MODEL_HF="org/repo:Q4")
        m, _ = settings.resolve("chat", s)
        self.assertEqual((m.mmproj, m.mmproj_url, m.no_mmproj), ("", "", False))
        self.assertNotIn("MMPROJ", settings.env_lines("chat", s, m))


class EnvLines(ModelFiles):
    def test_each_role_gets_its_own(self):
        s = self.with_dir()
        m, _ = settings.resolve("embed", s)
        text = settings.env_lines("embed", s, m)
        self.assertIn("ROLE=embed\n", text)
        self.assertIn("LUNA_HF=Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0\n", text)
        self.assertIn("EMBED_POOLING=last\n", text)
        self.assertNotIn("CTX=16384", text)

    def test_no_line_breaks(self):
        s = self.with_dir()
        s["EXTRA"] = "a\nb"
        m, _ = settings.resolve("chat", s)
        with self.assertRaises(settings.SettingsError):
            settings.env_lines("chat", s, m)


class DownloadSize(unittest.TestCase):
    TREE = [{"type": "file", "path": "README.md", "size": 100},
            {"type": "file", "path": "Qwen_Qwen3.5-9B-Q4_K_M.gguf", "size": 6_200_000_000},
            {"type": "file", "path": "Qwen_Qwen3.5-9B-Q8_0.gguf", "size": 9_800_000_000},
            {"type": "file", "path": "mmproj-Qwen_Qwen3.5-9B-f16.gguf", "size": 900_000_000},
            {"type": "file", "path": "big/Big-Q4_K_M-00001-of-00002.gguf", "size": 2},
            {"type": "file", "path": "big/Big-Q4_K_M-00002-of-00002.gguf", "size": 3}]

    def size(self, ref, tree=None, status=200):
        from unittest import mock

        from backends.containers import setup

        get = mock.Mock(return_value=mock.Mock(status_code=status, json=lambda: tree or self.TREE))
        return setup.hf_size(ref, get=get)

    def test_the_quant_asked_for(self):
        self.assertEqual(self.size("bartowski/Qwen_Qwen3.5-9B-GGUF:Q8_0"), 9_800_000_000)

    def test_default_quant_and_split_parts(self):
        tree = [f for f in self.TREE if f["path"].startswith("big/")]
        self.assertEqual(self.size("org/Big-GGUF", tree), 5)

    def test_unknown(self):
        self.assertIsNone(self.size("org/x:Q2_K"))
        self.assertIsNone(self.size("org/x:Q8_0", status=404))


if __name__ == "__main__":
    unittest.main()
