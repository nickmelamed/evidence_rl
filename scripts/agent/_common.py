"""Helpers shared by the agent hooks and checks.

Everything here uses the standard library only, so the hooks run with any
Python 3.9+ interpreter and need nothing installed.
"""
import fnmatch
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# Paths every repo protects, whatever .claude/protected-paths says. Removing a
# line from that file cannot switch these off.
BUILTIN_PROTECTED = [
    (".claude/settings.json", "agent permissions and hooks"),
    (".claude/settings.local.json", "local agent permissions and hooks"),
    (".claude/protected-paths", "this list"),
    (".claude/gate-commands", "the checks Claude must pass before stopping"),
    (".claude/style-ignore", "files the style checker skips"),
    (".claude/style-words", "the style checker's word list"),
    ("scripts/agent/*", "the checks that gate Claude's work"),
]

# Files that hold secrets. Example and template variants are safe to read.
SECRET_FILE = re.compile(
    r"(?<![\w.-])(?:"
    r"\.env(?:rc)?(?:\.[\w-]+)*"
    r"|\.netrc|\.pgpass|\.pypirc"
    r"|\.aws/credentials|\.ssh/id_[\w-]+(?!\.pub)"
    r"|\.config/gcloud/[\w./-]*credentials[\w.-]*"
    r")(?![\w-])"
)
SAFE_SECRET_SUFFIXES = {"example", "sample", "template", "dist", "defaults", "schema"}


def read_payload():
    """Return the hook payload from stdin, or an empty dict if there is none."""
    try:
        data = sys.stdin.read()
    except OSError:
        return {}
    if not data.strip():
        return {}
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def git(root, *args):
    """Run git in ``root`` and return stdout, or "" if the command fails."""
    try:
        r = subprocess.run(["git", *args], cwd=root, capture_output=True,
                           text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return r.stdout if r.returncode == 0 else ""


def project_root(payload=None):
    """Return the repo root, independent of where the session has cd'd to.

    Claude Code sets CLAUDE_PROJECT_DIR for hooks. The payload's ``cwd`` is
    the session's current directory, which can be a subdirectory, so it is
    only a fallback for finding the git top level.
    """
    env = os.environ.get("CLAUDE_PROJECT_DIR")
    if env and Path(env).is_dir():
        return Path(env).resolve()
    start = Path((payload or {}).get("cwd") or Path.cwd())
    top = git(start, "rev-parse", "--show-toplevel").strip()
    return Path(top).resolve() if top else start.resolve()


def relative_to_root(path, root, cwd=None):
    """Return ``path`` relative to ``root`` in POSIX form, or None if outside."""
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = Path(cwd or root) / p
    try:
        return p.resolve().relative_to(root).as_posix()
    except ValueError:
        return None


def glob_match(rel, pattern):
    """Match a repo-relative path against a protected-paths style glob.

    ``*`` crosses directory separators, a trailing ``/`` means everything
    under that directory, and a leading ``**/`` also matches at the root.
    """
    pattern = pattern.strip()
    if pattern.endswith("/"):
        pattern += "*"
    if fnmatch.fnmatchcase(rel, pattern):
        return True
    return pattern.startswith("**/") and fnmatch.fnmatchcase(rel, pattern[3:])


def load_protected(root):
    """Return (pattern, reason) pairs from the built-ins and the repo's list."""
    entries = list(BUILTIN_PROTECTED)
    f = root / ".claude" / "protected-paths"
    if f.is_file():
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            pattern, _, reason = line.partition("  #")
            entries.append((pattern.strip(), reason.strip()))
    return entries


def protected_reason(rel, root):
    """Return why ``rel`` is protected, or None if it is not."""
    for pattern, reason in load_protected(root):
        if glob_match(rel, pattern):
            return reason or "listed in .claude/protected-paths"
    return None


def secret_file_mentions(text):
    """Return the secret file names referenced in ``text``."""
    found = []
    for m in SECRET_FILE.finditer(text):
        name = m.group(0)
        suffix = name.rsplit(".", 1)[-1].lower() if name.count(".") > 1 else ""
        if suffix in SAFE_SECRET_SUFFIXES:
            continue
        found.append(name)
    return found


def base_ref(root):
    """Return the merge base with the default branch, or None."""
    candidates = []
    head = git(root, "symbolic-ref", "--quiet", "refs/remotes/origin/HEAD").strip()
    if head:
        candidates.append(head.replace("refs/remotes/", ""))
    candidates += ["origin/main", "main", "origin/master", "master"]
    for ref in candidates:
        base = git(root, "merge-base", "HEAD", ref).strip()
        if base:
            return base
    return None


def has_head(root):
    return bool(git(root, "rev-parse", "--verify", "--quiet", "HEAD").strip())
