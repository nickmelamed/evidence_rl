#!/usr/bin/env python3
"""Keep Claude working until the repo's fast checks pass.

Runs as a Claude Code Stop hook. When the working tree has changed since the
last check, it runs each command in ``.claude/gate-commands`` (one per line,
from the repo root) plus ``check_tests.py``. A failure exits with code 2,
which sends the output back to Claude and keeps the turn going.

After three failing rounds in a row it lets Claude stop and report, and it
stays quiet until the tree changes again. A turn that only answers a question
never reruns the checks, since nothing changed.

The whole run fits a time budget below the hook timeout in settings.json. A
command that exceeds it counts as a failure, because a gate that cannot
finish is not checking anything.

Set ``AGENT_GATE=off`` in the environment that launches Claude Code to turn
the gate off for a session, for example when the gate itself is broken.
"""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import git, has_head, project_root, read_payload  # noqa: E402
from check_tests import weakened_tests  # noqa: E402

MAX_ROUNDS = 3
BUDGET_S = 1080  # stays under the 1200 s hook timeout in settings.json
PER_COMMAND_S = 900
BIG_FILE = 1 << 20
STATE = ".claude/.gate-state.json"
DEFAULT_COMMANDS = ["python3 scripts/agent/check_style.py --changed ."]


def fingerprint(root):
    h = hashlib.sha256()
    if has_head(root):
        h.update(git(root, "rev-parse", "HEAD").encode())
        h.update(git(root, "diff", "HEAD", "--binary").encode())
    else:
        h.update(git(root, "diff", "--cached", "--binary").encode())
        h.update(git(root, "diff", "--binary").encode())
    for p in git(root, "ls-files", "--others", "--exclude-standard").splitlines():
        if p == STATE:
            continue
        f = root / p
        h.update(p.encode())
        try:
            st = f.stat()
            if st.st_size > BIG_FILE:
                h.update(f"{st.st_size}:{st.st_mtime_ns}".encode())
            else:
                h.update(f.read_bytes())
        except OSError:
            pass
    return h.hexdigest()


def load_commands(root):
    f = root / ".claude" / "gate-commands"
    if not f.is_file():
        return DEFAULT_COMMANDS
    return [line.strip() for line in f.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")]


def run_checks(root):
    failures = []
    started = time.monotonic()
    for cmd in load_commands(root):
        remaining = BUDGET_S - (time.monotonic() - started)
        if remaining < 5:
            failures.append(f"$ {cmd}\nnot run, the gate's time budget "
                            f"({BUDGET_S}s) is used up. Ask the owner to move "
                            "slow checks out of .claude/gate-commands.")
            continue
        timeout = min(PER_COMMAND_S, remaining)
        try:
            r = subprocess.run(cmd, shell=True, cwd=root, capture_output=True,
                               text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            failures.append(f"$ {cmd}\ntimed out after {timeout:.0f}s")
            continue
        if r.returncode == 0:
            continue
        tail = (r.stdout + r.stderr).strip().splitlines()[-40:]
        note = ""
        if r.returncode == 127:
            note = ("\nThe command was not found. Install the tool, or ask the "
                    "owner to change .claude/gate-commands.")
        failures.append(f"$ {cmd}\n" + "\n".join(tail) + note)
    weak = weakened_tests(root)
    if weak:
        failures.append("Tests were weakened compared with the base branch:\n"
                        + "\n".join(weak)
                        + "\nRestore them, or stop and ask the owner to approve.")
    return "\n\n".join(failures)


def save(path, state):
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(state), encoding="utf-8")


def main():
    if os.environ.get("AGENT_GATE", "").lower() in ("off", "0", "false", "no"):
        return 0
    payload = read_payload()
    root = project_root(payload)
    if not git(root, "rev-parse", "--is-inside-work-tree").strip():
        return 0

    state_file = root / STATE
    try:
        state = json.loads(state_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        state = {}

    current = fingerprint(root)
    if state.get("checked") == current:
        if state.get("ok") or state.get("released"):
            return 0
        failures = state.get("failures", "")
    else:
        failures = run_checks(root)
        current = fingerprint(root)
        if not failures:
            save(state_file, {"checked": current, "ok": True})
            return 0

    live_streak = not (state.get("ok") or state.get("released"))
    rounds = (state.get("rounds", 0) if live_streak else 0) + 1
    if rounds > MAX_ROUNDS:
        save(state_file, {"checked": current, "ok": False, "released": True,
                          "failures": failures})
        print(json.dumps({"systemMessage": (
            f"Stop gate: checks still fail after {MAX_ROUNDS} rounds, so Claude "
            "was allowed to stop. They run again after the next change.")}))
        return 0
    save(state_file, {"checked": current, "ok": False, "released": False,
                      "rounds": rounds, "failures": failures})

    msg = failures
    if rounds == MAX_ROUNDS:
        msg += ("\n\nThis is the last automatic round. If you cannot fix this, "
                "stop and tell the owner exactly what is failing and why.")
    msg += ("\n\nIf a check itself looks wrong, say so and stop. Do not weaken "
            "the check or edit .claude/gate-commands.")
    print(f"Definition of done not met (round {rounds} of {MAX_ROUNDS}).\n\n{msg}",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
