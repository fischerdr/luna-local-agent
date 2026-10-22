"""The scope fence for feat/containers.

What the branch may add or change (plans/container-migration-plan-v2.md,
section 2.2), compared with the commit it branched from.

    python tests/check_scope_fence.py              (committed work only)
    python tests/check_scope_fence.py --worktree   (plus uncommitted changes)

No test_ prefix on purpose: unittest discovery skips it, since after a merge
it would only be comparing main with itself. Run it at every phase gate.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Added files: anything under these folders, or these exact files.
ADDED_DIRS = ("services/", "deploy/", "tests/", "plans/")
ADDED_FILES = {"docs/containers.md", "readme-containers.md", ".flake8"}

# Modified files: opt-in edits only.
MODIFIED = {
    "installer/install.py", "launch.py", "config.example.json", ".gitignore",
    "embedmem.py",          # P9's default-off on-demand hook, approved 2026-10-22
    "README.md", "docs/README.md", "docs/_Sidebar.md", "docs/install.md",
    "docs/configuration.md", "docs/development.md", "docs/knowledge.md",
    "docs/vision.md", "docs/reasoning.md", "docs/sysadmin.md",
}

# The one docs line that may change (section 11): one line out, one in.
DELETIONS_ALLOWED = {"docs/knowledge.md": 1}

# Written at runtime, never tracked: main already gitignores both.
RUNTIME = ("llama/.api_key", "llama/server.local.env")


def git(*args):
    """Run git in the repo and return its output."""
    return subprocess.run(["git", *args], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout


def lines(*args):
    """Git output as a list of non-empty lines."""
    return [ln for ln in git(*args).splitlines() if ln]


def added_ok(path):
    """Whether a new file is somewhere the branch may add one."""
    return path in ADDED_FILES or path.startswith(ADDED_DIRS)


def check(worktree=False):
    """Every broken rule, one line each. Empty means the fence holds."""
    # FENCE_BASE=6d1ef9f once the branch lives in a repo with no such main.
    base = os.environ.get("FENCE_BASE") or git("merge-base", "HEAD", "main").strip()
    # Committed work by default; with --worktree the diff is against the files
    # on disk, so staged and unstaged edits count too.
    rng = [base] if worktree else [base, "HEAD"]
    bad = []

    # 1. Modified, deleted or renamed: only the opt-in files.
    for p in lines("diff", "--name-only", "--diff-filter=MDR", *rng):
        if p not in MODIFIED:
            bad.append(f"changed outside the fence: {p}")

    # 2. Added files: only under the allowed paths.
    for p in lines("diff", "--name-only", "--diff-filter=A", *rng):
        if not added_ok(p):
            bad.append(f"added outside the fence: {p}")
    if worktree:
        for p in lines("ls-files", "--others", "--exclude-standard"):
            if not added_ok(p):
                bad.append(f"untracked outside the fence: {p}")

    # 3. Docs are additive: no deleted lines, bar the one knowledge.md phrase.
    for row in lines("diff", "--numstat", *rng, "--", "docs", "README.md"):
        _, deleted, p = row.split("\t", 2)
        if deleted != "-" and int(deleted) > DELETIONS_ALLOWED.get(p, 0):
            bad.append(f"{p}: {deleted} line(s) deleted (docs are additive)")

    # 4. The runtime files stay out of git.
    for p in RUNTIME:
        if lines("ls-files", p):
            bad.append(f"runtime file is tracked: {p}")
        elif subprocess.run(["git", "check-ignore", "-q", p], cwd=ROOT).returncode:
            bad.append(f"runtime file isn't gitignored: {p}")
    return bad


def main():
    """Print the verdict; exit 1 when the fence is broken."""
    bad = check(worktree="--worktree" in sys.argv[1:])
    for b in bad:
        print("✗", b)
    if not bad:
        print("✓ scope fence holds")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
