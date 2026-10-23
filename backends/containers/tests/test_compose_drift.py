"""deploy/ against python -m backends.containers (services/run.py, settings.py,
images.py): the same roles, ports, settings, secrets, volumes and pinned
images, so a compose server and local containers can't drift apart.

There's no PyYAML on the test machines, so a small reader for the YAML
these files use - mappings, lists, [flow, lists], quoted strings - is here.

    python -m unittest discover tests        (from the repo folder)
"""
import glob
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from tests.offline import FAKE  # noqa: E402,F401  (first: no real servers)

from backends.containers import VOLUMES, images, run, settings  # noqa: E402

DEPLOY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "deploy")


def scalar(text):
    text = text.strip()
    if text[:1] == "[" and text[-1:] == "]":
        return [scalar(x) for x in re.findall(r'"[^"]*"|\'[^\']*\'|[^,\s][^,]*', text[1:-1])]
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1]
    return text


def strip_comment(line):
    out, quote = [], None
    for i, ch in enumerate(line):
        if quote:
            quote = None if ch == quote else quote
        elif ch in "\"'":
            quote = ch
        elif ch == "#" and (i == 0 or line[i - 1] == " "):
            break
        out.append(ch)
    return "".join(out).rstrip()


def parse(text):
    """The YAML subset compose.yaml uses, as dicts and lists."""
    lines = [(len(ln) - len(ln.lstrip(" ")), ln.strip()) for ln in map(strip_comment, text.splitlines())
             if ln.strip()]

    def block(i, indent):
        if lines[i][1].startswith("- "):
            out = []
            while i < len(lines) and lines[i][0] == indent and lines[i][1].startswith("- "):
                item = lines[i][1][2:]
                if re.match(r"^[\w.-]+:( |$)", item):
                    # a mapping inside a list: its first key is on the dash's line
                    lines[i] = (indent + 2, item)
                    value, i = block(i, indent + 2)
                    out.append(value)
                else:
                    out.append(scalar(item))
                    i += 1
            return out, i
        out = {}
        while i < len(lines) and lines[i][0] == indent:
            key, _, rest = lines[i][1].partition(":")
            if rest.strip():
                out[key.strip()] = scalar(rest)
                i += 1
            elif i + 1 < len(lines) and lines[i + 1][0] > indent:
                out[key.strip()], i = block(i + 1, lines[i + 1][0])
            else:
                out[key.strip()] = None
                i += 1
        return out, i

    return block(0, 0)[0]


def load(name):
    with open(os.path.join(DEPLOY, name)) as f:
        return parse(f.read())


def default(value):
    """${VAR:-default} -> default; a plain value as it is."""
    m = re.fullmatch(r"\$\{\w+:-(.*)\}", value)
    return m.group(1) if m else value


ROLES = {"llama": "chat", "embed": "embed", "rerank": "rerank"}


class Reader(unittest.TestCase):
    def test_reader(self):
        doc = parse('a:\n  b: "x # y"   # c\n  l:\n    - 1\n    - k: v\n      m: [p, "q r"]\n')
        self.assertEqual(doc, {"a": {"b": "x # y", "l": ["1", {"k": "v", "m": ["p", "q r"]}]}})


class Drift(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = load("compose.yaml")
        cls.svc = cls.base["services"]

    def test_the_four_services(self):
        self.assertEqual(set(self.svc), {"llama", "embed", "rerank", "tts"})

    def test_roles_and_their_settings(self):
        for name, role in ROLES.items():
            with self.subTest(service=name):
                env = self.svc[name]["environment"]
                self.assertEqual(env["ROLE"], role)
                self.assertIn("LUNA_HF", env)
                self.assertEqual(set(settings.PASS[role]) - set(env), set(), "settings run_args passes")

    def test_side_model_defaults_match_services_env(self):
        with open(os.path.join(settings.HERE, "services.env")) as f:
            text = f.read()
        for name, key in (("embed", "EMBED_HF"), ("rerank", "RERANK_HF")):
            want = re.search(rf'^{key}="([^"]*)"', text, re.M).group(1)
            self.assertEqual(default(self.svc[name]["environment"]["LUNA_HF"]), want)

    def test_ports(self):
        want = {"llama": ("CHAT_PORT", 8080, 8080), "embed": ("EMBED_PORT", 8081, 8080),
                "rerank": ("RERANK_PORT", 8082, 8080), "tts": ("TTS_PORT", 8899, 8899)}
        s = dict(settings.DEFAULTS)
        for name, (var, host, inner) in want.items():
            with self.subTest(service=name):
                (port,) = self.svc[name]["ports"]
                self.assertRegex(port, r"^\$\{LUNA_BIND:\?[^}]+\}:")
                self.assertTrue(port.endswith(f":${{{var}:-{host}}}:{inner}"), port)
                role = ROLES.get(name, "tts")
                self.assertEqual(settings.port(role, s), host)
                if name != "tts":
                    self.assertEqual(run.publish("127.0.0.1", host).split(":")[-1], str(inner))

    def test_secrets_at_the_same_paths(self):
        for name in ROLES:
            targets = {s["target"] for s in self.svc[name]["secrets"]}
            self.assertEqual(targets, {"api_key", "hf_token"})
        self.assertNotIn("secrets", self.svc["tts"])          # the voice takes no key
        self.assertEqual(set(self.base["secrets"]), {"api_key", "hf_token"})

    def test_volumes(self):
        self.assertIn("models:/models", self.svc["llama"]["volumes"])
        self.assertIn("data:/data", self.svc["llama"]["volumes"])
        self.assertIn("tts-cache:/cache", self.svc["tts"]["volumes"])
        names = {default(v["name"].replace("${LUNA_VOLUMES:-luna}", "luna")) for v in self.base["volumes"].values()}
        self.assertEqual(names, set(VOLUMES))

    def test_images_pinned_like_services(self):
        for name in ("llama", "rerank"):
            args = self.svc[name]["build"]["args"]
            self.assertEqual(self.svc[name]["build"]["context"], "../llama")
            self.assertEqual(default(args["BACKEND"]), images.BASES["cpu"][0])
            self.assertEqual(default(args["BASE_DIGEST"]), images.BASES["cpu"][1])
        embed = self.svc["embed"]["build"]["args"]
        self.assertEqual((embed["BACKEND"], embed["BASE_DIGEST"]), images.BASES["cpu"])
        self.assertEqual(default(self.svc["tts"]["build"]["args"]["TORCH_INDEX"]), images.TORCH_INDEX["cpu"])

    def test_gpu_files_pinned_like_services(self):
        for name, backend in (("compose.cuda.yaml", "cuda13"), ("compose.cuda.docker.yaml", "cuda13"),
                              ("compose.rocm.yaml", "rocm"), ("compose.vulkan.yaml", "vulkan")):
            svc = load(name)["services"]
            with self.subTest(file=name):
                self.assertNotIn("embed", svc)                  # the embedder stays on the CPU
                for role in ("llama", "rerank"):
                    args = svc[role]["build"]["args"]
                    self.assertEqual(default(args["BACKEND"]), images.BASES[backend][0])
                    self.assertEqual(default(args["BASE_DIGEST"]), images.BASES[backend][1])
                if backend == "cuda13":
                    self.assertEqual(default(svc["tts"]["build"]["args"]["TORCH_INDEX"]), images.TORCH_INDEX["cuda13"])

    def test_gpu_flags_like_run_args(self):
        podman_cuda = load("compose.cuda.yaml")["services"]["llama"]
        self.assertEqual(podman_cuda["devices"], ["nvidia.com/gpu=all"])
        self.assertIn("label=disable", podman_cuda["security_opt"])
        rocm = load("compose.rocm.yaml")["services"]["llama"]
        flags = run.gpu_flags("rocm", selinux=True, render=False)
        self.assertEqual(rocm["devices"], [flags[i + 1] for i, f in enumerate(flags) if f == "--device"])
        self.assertIn("seccomp=unconfined", rocm["security_opt"])
        docker = load("compose.cuda.docker.yaml")["services"]["llama"]["deploy"]["resources"]["reservations"]
        self.assertEqual(docker["devices"][0]["driver"], "nvidia")

    def test_health_like_run_args(self):
        test = self.svc["llama"]["healthcheck"]["test"]
        self.assertEqual(" ".join(test[1:]), run.HEALTH)

    def test_never_host_network(self):
        for path in glob.glob(os.path.join(DEPLOY, "compose*.yaml")):
            with open(path) as f:
                self.assertNotIn("network_mode", f.read(), path)


if __name__ == "__main__":
    unittest.main()
