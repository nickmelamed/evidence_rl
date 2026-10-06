#!/usr/bin/env python3
"""Give Claude the current plan and gate status at session start and after compaction.

Runs as a Claude Code SessionStart hook (matchers ``startup`` and
``compact``). Its output is added to Claude's context. It prints the "Now"
section of PROGRESS.md, the branch and uncommitted files, and the last stop
gate result, so this state survives compaction without a standing
instruction asking Claude to remember it.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import git, project_root, read_payload  # noqa: E402

MAX_LINES = 20


def progress_now(root):
    f = root / "PROGRESS.md"
    if not f.is_file():
        return []
    text = f.read_text(encoding="utf-8")
    m = re.search(r"^##\s+Now\s*$(.*?)(?=^##\s|\Z)", text, re.M | re.S)
    if not m:
        return []
    lines = [line for line in m.group(1).strip().splitlines() if line.strip()]
    return lines[:MAX_LINES]


def gate_status(root):
    try:
        state = json.loads((root / ".claude" / ".gate-state.json").read_text())
    except (OSError, ValueError):
        return "not run yet"
    if state.get("ok"):
        return "passing at the last check"
    first = (state.get("failures") or "").strip().splitlines()[:8]
    label = "failing (Claude was released after 3 rounds)" if state.get("released") \
        else "failing"
    return label + ("\n  " + "\n  ".join(first) if first else "")


def main():
    payload = read_payload()
    root = project_root(payload)
    out = [f"Session context ({payload.get('source') or 'start'})."]
    now = progress_now(root)
    if now:
        out.append("Current phase from PROGRESS.md:")
        out += [f"  {line.strip()}" for line in now]
    branch = git(root, "branch", "--show-current").strip()
    if branch:
        out.append(f"Branch: {branch}")
    changed = git(root, "status", "--porcelain").splitlines()
    if changed:
        shown = [c[3:] for c in changed[:MAX_LINES]]
        extra = len(changed) - len(shown)
        out.append("Uncommitted: " + ", ".join(shown)
                   + (f", and {extra} more" if extra > 0 else ""))
    out.append("Stop gate: " + gate_status(root))
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"session_context.py failed ({exc!r}).", file=sys.stderr)
        sys.exit(1)
