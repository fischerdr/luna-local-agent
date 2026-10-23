"""main.py's functions read the config module imported at the top. An
`import config` inside one of them makes config local to the whole function,
so every use before that line fails - /settings crashed with "cannot access
local variable 'config'" because /camera, further down handle_input,
imported it again. Read from the source; main.py isn't imported.

    python -m unittest discover tests        (from the repo folder)
"""
import ast
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.offline import FAKE  # noqa: E402,F401  (first: no real servers)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ConfigScope(unittest.TestCase):
    def test_no_function_binds_config_again(self):
        with open(os.path.join(ROOT, "main.py"), encoding="utf-8") as f:
            tree = ast.parse(f.read())
        rebound = []
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(fn):
                names = []
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    names = [a.asname or a.name.split(".")[0] for a in node.names]
                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    names = [node.id]
                if "config" in names:
                    rebound.append(f"{fn.name}, line {node.lineno}")
        self.assertEqual(rebound, [])


if __name__ == "__main__":
    unittest.main()
