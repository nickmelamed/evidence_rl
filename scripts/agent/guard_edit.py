#!/usr/bin/env python3
"""Ask the owner before Claude edits a protected file.

Runs as a Claude Code PreToolUse hook on Edit, Write, MultiEdit, and
NotebookEdit. Protected paths are the built-ins in ``_common.py`` plus the
globs in ``.claude/protected-paths`` (one per line, relative to the repo
root, text after two spaces and ``#`` is the reason). A match gets a
permission prompt that names the reason.

Paths are resolved against the project root, not the session's current
directory, so a ``cd`` into a subdirectory does not change what matches.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import project_root, protected_reason, read_payload, relative_to_root  # noqa: E402

PATH_FIELDS = ("file_path", "notebook_path", "path")


def main():
    payload = read_payload()
    tool_input = payload.get("tool_input") or {}
    target = next((tool_input[k] for k in PATH_FIELDS if tool_input.get(k)), None)
    if not target:
        return 0
    root = project_root(payload)
    rel = relative_to_root(target, root, payload.get("cwd"))
    if rel is None:
        return 0
    reason = protected_reason(rel, root)
    if reason:
        print(json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "ask",
                "permissionDecisionReason": f"{rel} is protected ({reason}).",
            }
        }))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"guard_edit.py failed ({exc!r}).", file=sys.stderr)
        sys.exit(1)
