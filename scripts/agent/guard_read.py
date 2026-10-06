#!/usr/bin/env python3
"""Keep Claude's file tools away from secret files.

Runs as a Claude Code PreToolUse hook on Read and Grep. Glob only lists
names, so it is not checked. The deny rules
in settings.json cover ``.env`` itself. This hook covers the variants
(``.env.production``, ``.envrc``, cloud credentials) while still allowing
``.env.example`` and other templates, which a deny rule cannot carve out.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import read_payload, secret_file_mentions  # noqa: E402


def main():
    payload = read_payload()
    tool_input = payload.get("tool_input") or {}
    targets = [str(tool_input[k]) for k in ("file_path", "path", "glob")
               if tool_input.get(k)]
    for target in targets:
        found = secret_file_mentions("/" + target.lstrip("/"))
        if found:
            print(f"Blocked: {found[0]} holds secrets and stays out of the "
                  "conversation. Ask the owner for the variable name you need.",
                  file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"guard_read.py failed ({exc!r}).", file=sys.stderr)
        sys.exit(1)
