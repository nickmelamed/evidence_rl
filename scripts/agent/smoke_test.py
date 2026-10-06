#!/usr/bin/env python3
"""Check that this repo's agent hooks are wired up and behave as expected.

Run from the repo root after installing or changing the tooling::

    python3 scripts/agent/smoke_test.py          # hooks and settings
    python3 scripts/agent/smoke_test.py --gate   # also run the stop gate once

It feeds sample payloads to each hook and checks the outcome (allow, ask, or
block), checks that settings.json points every hook at a script that exists,
and tries an edit to each literal path in ``.claude/protected-paths``.
Nothing in the repo is changed except a temporary file that is removed.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _common import project_root  # noqa: E402

EXPECTED_HOOKS = {
    ("PreToolUse", "Bash"): "guard_bash.py",
    ("PreToolUse", "Edit"): "guard_edit.py",
    ("PreToolUse", "NotebookEdit"): "guard_edit.py",
    ("PreToolUse", "Read"): "guard_read.py",
    ("PostToolUse", "Edit"): "check_style.py",
    ("Stop", ""): "stop_gate.py",
    ("SessionStart", "compact"): "session_context.py",
}


def run_hook(script, payload, root, env_extra=None):
    env = dict(os.environ, CLAUDE_PROJECT_DIR=str(root), **(env_extra or {}))
    args = [sys.executable, str(HERE / script)]
    if script == "check_style.py":
        args.append("--hook")
    r = subprocess.run(args, input=json.dumps(payload), capture_output=True,
                       text=True, cwd=root, env=env, timeout=1200)
    if r.returncode == 2:
        return "block", r.stderr.strip()
    if '"ask"' in r.stdout:
        return "ask", r.stdout.strip()
    if r.returncode == 0:
        return "allow", ""
    return f"error {r.returncode}", (r.stderr or r.stdout).strip()


def bash(cmd, root):
    return {"tool_name": "Bash", "tool_input": {"command": cmd}, "cwd": str(root)}


def edit(path, root, tool="Edit"):
    field = "notebook_path" if tool == "NotebookEdit" else "file_path"
    return {"tool_name": tool, "tool_input": {field: str(root / path)},
            "cwd": str(root)}


def check_settings(root):
    problems = []
    f = root / ".claude" / "settings.json"
    try:
        settings = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"settings.json unreadable: {exc}"]
    hooks = settings.get("hooks", {})
    for (event, tool), script in EXPECTED_HOOKS.items():
        found = False
        for group in hooks.get(event, []):
            matcher = group.get("matcher", "")
            if tool and tool not in matcher.split("|"):
                continue
            for h in group.get("hooks", []):
                if script in h.get("command", ""):
                    found = True
        if not found:
            where = f"{event} {tool}".strip()
            problems.append(f"settings.json has no {where} hook running {script}")
    for event, groups in hooks.items():
        for group in groups:
            for h in group.get("hooks", []):
                cmd = h.get("command", "")
                for part in cmd.replace('"', " ").split():
                    if part.endswith(".py") and "scripts/agent/" in part:
                        rel = part.split("scripts/agent/", 1)[1]
                        if not (root / "scripts" / "agent" / rel).is_file():
                            problems.append(f"{event} hook points at missing {part}")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gate", action="store_true",
                        help="also run the stop gate once and show its result")
    args = parser.parse_args()
    root = project_root()

    cases = [
        ("guard_bash.py", bash("git status", root), "allow"),
        ("guard_bash.py", bash("git push --force origin main", root), "block"),
        ("guard_bash.py", bash("git -C . commit --no-verify -m x", root), "block"),
        ("guard_bash.py", bash("SKIP=check-style git commit -m x", root), "block"),
        ("guard_bash.py", bash("cat .env.development", root), "block"),
        ("guard_bash.py", bash("printenv", root), "block"),
        ("guard_bash.py", bash("git reset --hard HEAD~1", root), "ask"),
        ("guard_bash.py", bash("echo x > .claude/gate-commands", root), "ask"),
        ("guard_edit.py", edit(".claude/gate-commands", root), "ask"),
        ("guard_edit.py", edit("scripts/agent/stop_gate.py", root), "ask"),
        ("guard_edit.py", edit("scripts/agent/x.ipynb", root, "NotebookEdit"), "ask"),
        ("guard_read.py", {"tool_name": "Read",
                           "tool_input": {"file_path": str(root / ".env.production")}},
         "block"),
        ("guard_read.py", {"tool_name": "Read",
                           "tool_input": {"file_path": str(root / ".env.example")}},
         "allow"),
    ]
    protected = root / ".claude" / "protected-paths"
    if protected.is_file():
        for line in protected.read_text(encoding="utf-8").splitlines():
            pattern = line.partition("  #")[0].strip()
            if pattern and not pattern.startswith("#") and not set("*?[") & set(pattern):
                if pattern.endswith("/"):
                    pattern += "example.md"
                cases.append(("guard_edit.py", edit(pattern, root), "ask"))

    probe = root / "_smoke_style_probe.md"
    probe.write_text("This line has an em dash — in it.\n", encoding="utf-8")
    cases.append(("check_style.py", edit(probe.name, root, "Write"), "block"))

    failures = 0
    try:
        for script, payload, want in cases:
            got, detail = run_hook(script, payload, root)
            what = payload["tool_input"].get("command") or next(
                iter(payload["tool_input"].values()))
            what = str(what).replace(str(root) + "/", "")
            ok = got == want
            failures += not ok
            print(f"{'ok  ' if ok else 'FAIL'} {script:<18} {want:<6} {what}")
            if not ok:
                print(f"     got {got}: {detail[:300]}")
    finally:
        probe.unlink(missing_ok=True)

    got, _ = run_hook("stop_gate.py", {"cwd": str(root)}, root, {"AGENT_GATE": "off"})
    ok = got == "allow"
    failures += not ok
    print(f"{'ok  ' if ok else 'FAIL'} stop_gate.py       allow  AGENT_GATE=off")

    got, detail = run_hook("session_context.py", {"source": "compact"}, root)
    ok = got == "allow"
    failures += not ok
    print(f"{'ok  ' if ok else 'FAIL'} session_context.py allow  compact")

    for problem in check_settings(root):
        failures += 1
        print(f"FAIL settings: {problem}")

    if args.gate:
        got, detail = run_hook("stop_gate.py", {"cwd": str(root)}, root)
        print(f"\nStop gate result: {got}")
        if detail:
            print(detail[:2000])

    print(f"\n{'All hooks behave as expected.' if not failures else f'{failures} problem(s).'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
