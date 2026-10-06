#!/usr/bin/env python3
"""Stop shell commands that bypass checks, destroy work, or expose secrets.

Runs as a Claude Code PreToolUse hook on Bash. The command is split into
simple commands the way a shell would (``&&``, ``;``, pipes, subshells,
``bash -c`` strings), so flag order, short-flag clusters, and git's global
options do not hide anything.

Two outcomes besides allowing the command.

Block (exit 2). Never allowed from a session, whoever asks. Skipping commit
hooks, force-pushing, reading or printing secrets, deleting the repo.

Ask (permissionDecision "ask"). The owner decides in the permission prompt.
Discarding uncommitted work, deleting branches or stashes, recursive deletes,
and shell writes to paths in ``.claude/protected-paths``.

This is defense in depth next to the permission rules in settings.json. A
determined program (a Python script that opens ``.env``) can still get past
it, which is why secrets should also stay out of the working tree.
"""
import json
import os
import re
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    project_root,
    protected_reason,
    read_payload,
    relative_to_root,
    secret_file_mentions,
)

PUNCT = set("();<>|&")
WRAPPERS = {"sudo", "command", "builtin", "nice", "nohup", "time", "timeout",
            "stdbuf", "noglob", "exec"}
SHELLS = {"sh", "bash", "zsh", "dash", "ksh"}
HOOK_BYPASS_ENV = re.compile(
    r"^(SKIP|HUSKY|PRE_COMMIT_ALLOW_NO_CONFIG|GIT_CONFIG_COUNT|"
    r"GIT_CONFIG_PARAMETERS|GIT_CONFIG_KEY_\d+|GIT_CONFIG_VALUE_\d+)="
)
SECRET_NAME = r"\w*(API_?KEY|SECRET|TOKEN|PASSW(OR)?D|CREDENTIAL|PRIVATE_KEY)\w*"
SECRET_VAR = re.compile(r"\$\{?" + SECRET_NAME, re.I)
SECRET_VAR_NAME = re.compile(r"^" + SECRET_NAME + r"$", re.I)
DISPOSABLE = {
    "build", "dist", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", ".hypothesis", "htmlcov", ".coverage", ".tox", ".nox",
    "node_modules", ".ipynb_checkpoints", ".quarto", "_site", "site",
    ".gate-state.json",
}
CATASTROPHIC = {"/", "~", "~/", "$HOME", "${HOME}", ".", "./", "..", "../",
                "*", "./*", ".git", "./.git"}
IGNORE_FILES = (".gitignore", ".dockerignore", ".git/info/exclude")

MSG_NO_VERIFY = "Never bypass hooks with --no-verify. Fix what the hook reports."
MSG_COMMIT_N = "git commit -n skips hooks. Fix what the hook reports instead."
MSG_HOOKSPATH = "Changing core.hooksPath disables commit hooks."
MSG_HOOK_ENV = "{} skips commit hooks. Fix what the hook reports instead."
MSG_FORCE = "Force-pushing is not allowed. Ask the owner if history must change."
MSG_SECRET_READ = ("Do not read {}. Secrets stay out of the conversation. "
                   "Ask the owner for the variable name you need.")
MSG_SECRET_ADD = "Never stage {}. Secret files stay out of git."
MSG_SECRET_PRINT = "Do not print secrets or the environment."


HEREDOC = re.compile(
    r"<<-?[ \t]*(['\"]?)([A-Za-z_]\w*)\1([^\n]*)\n(.*?)\n[ \t]*\2[ \t]*(?=\n|$)", re.S)
TEXT_CONSUMERS = {"cat", "tee"}


def strip_text_heredocs(command):
    """Drop heredoc bodies that only feed text to cat or tee.

    Such a body is data, such as a document or a commit message that happens
    to name a secret file. Bodies handed to an interpreter, a shell, or a
    pipe stay in, because they can run.
    """
    def drop(m):
        before = re.split(r"[;&|\n(`]", command[:m.start()])[-1].split()
        words = [w for w in before if not re.match(r"^[A-Za-z_]\w*=", w)]
        consumer = os.path.basename(words[0]) if words else ""
        if consumer in TEXT_CONSUMERS and "|" not in m.group(3):
            return command[m.start():m.start(4)].rstrip("\n") + "\n" + m.group(2)
        return m.group(0)
    return HEREDOC.sub(drop, command)


def tokenize(command):
    lex = shlex.shlex(command.replace("\n", " ; "), posix=True,
                      punctuation_chars=True)
    lex.whitespace_split = True
    lex.commenters = ""
    try:
        return list(lex)
    except ValueError:
        return command.replace("\n", " ; ").split()


def split_segments(tokens):
    """Group tokens into simple commands with their redirect targets."""
    segs, cur, pending = [], {"words": [], "out": [], "in": []}, None
    for tok in tokens:
        if tok and set(tok) <= PUNCT:
            if "(" in tok or ")" in tok or not ({"<", ">"} & set(tok)):
                segs.append(cur)
                cur, pending = {"words": [], "out": [], "in": []}, None
                continue
            if cur["words"] and cur["words"][-1] in {"0", "1", "2"}:
                cur["words"].pop()
            if tok.endswith(">&"):
                pending = "fd"
            else:
                pending = "out" if ">" in tok else "in"
            continue
        if pending == "fd":
            if not (tok.isdigit() or tok == "-"):
                cur["out"].append(tok)
        elif pending in ("out", "in"):
            cur[pending].append(tok)
        else:
            cur["words"].append(tok.strip("`"))
        pending = None
    segs.append(cur)
    return [s for s in segs if s["words"] or s["out"] or s["in"]]


def strip_wrappers(words):
    """Drop env assignments and wrappers such as sudo, env, timeout, xargs."""
    assigns, i = [], 0
    while i < len(words):
        w = words[i]
        base = os.path.basename(w)
        if re.match(r"^[A-Za-z_]\w*=", w):
            assigns.append(w)
            i += 1
        elif base == "env" and i + 1 < len(words):
            i += 1
            while i < len(words) and words[i].startswith("-"):
                i += 2 if words[i] in ("-u", "--unset", "-C", "--chdir") else 1
        elif base in WRAPPERS and i + 1 < len(words):
            i += 1
            while i < len(words) and words[i].startswith("-"):
                i += 2 if words[i] in ("-u", "-g", "-n", "-k", "-s") else 1
            if base == "timeout" and i < len(words):
                i += 1
        elif base == "xargs" and i + 1 < len(words):
            i += 1
            while i < len(words) and words[i].startswith("-"):
                takes = words[i] in ("-n", "-I", "-L", "-P", "-d", "-E", "-s")
                i += 2 if takes else 1
        else:
            break
    return assigns, words[i:]


def short_flag(args, letter, takes_arg="", optional_arg="", long_arg=()):
    """Return True if a short option ``letter`` appears, even in a cluster."""
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--":
            return False
        if a.startswith("--"):
            i += 2 if a in long_arg else 1
            continue
        if a.startswith("-") and len(a) > 1:
            letters = a[1:]
            for j, ch in enumerate(letters):
                if ch == letter:
                    return True
                if ch in takes_arg:
                    if j == len(letters) - 1:
                        i += 1
                    break
                if ch in optional_arg:
                    break
        i += 1
    return False


def positionals(args, takes_arg="", long_arg=()):
    out, i, after_dd = [], 0, False
    while i < len(args):
        a = args[i]
        if after_dd:
            out.append(a)
        elif a == "--":
            after_dd = True
        elif a.startswith("--"):
            i += 1 if (a not in long_arg or "=" in a) else 2
            continue
        elif a.startswith("-") and len(a) > 1:
            if a[-1] in takes_arg:
                i += 2
                continue
        else:
            out.append(a)
        i += 1
    return out


COMMIT_LONG = ("--message", "--file", "--author", "--date", "--reuse-message",
               "--reedit-message", "--fixup", "--squash", "--template",
               "--cleanup", "--trailer", "--pathspec-from-file")
GIT_GLOBAL_ARG = {"-C", "-c", "--git-dir", "--work-tree", "--namespace",
                  "--super-prefix", "--config-env", "--exec-path"}
WIDE_PATHSPEC = {".", "./", ":/", ":/*", "*", ":(top)", ":(glob)**"}


def analyze_git(args):
    """Return (blocks, asks) for one git invocation's arguments."""
    blocks, asks = [], []
    i = 0
    while i < len(args):
        a = args[i]
        if a in GIT_GLOBAL_ARG:
            val = args[i + 1] if i + 1 < len(args) else ""
            if a in ("-c", "--config-env") and "hookspath" in val.lower():
                blocks.append(MSG_HOOKSPATH)
            i += 2
            continue
        if a.startswith("-"):
            if "hookspath" in a.lower():
                blocks.append(MSG_HOOKSPATH)
            i += 1
            continue
        break
    if i >= len(args):
        return blocks, asks
    sub, rest = args[i], args[i + 1:]

    if any(r == "--no-verify" or r.startswith("--no-verify=") for r in rest):
        blocks.append(MSG_NO_VERIFY)
    if sub == "commit":
        if short_flag(rest, "n", "mFCct", "Su", COMMIT_LONG):
            blocks.append(MSG_COMMIT_N)
    elif sub == "push":
        pos = positionals(rest, "o", ("--repo", "--push-option", "--receive-pack",
                                      "--exec"))
        if (any(r.split("=")[0] in ("--force", "--force-with-lease",
                                    "--force-if-includes", "--mirror")
                for r in rest)
                or short_flag(rest, "f", "o")
                or any(p.startswith("+") for p in pos)):
            blocks.append(MSG_FORCE)
    elif sub == "config":
        if any("hookspath" in r.lower() for r in rest):
            blocks.append(MSG_HOOKSPATH)
    elif sub in ("filter-branch", "filter-repo"):
        blocks.append("Rewriting history is not allowed from a session.")
    elif sub == "reset":
        if "--hard" in rest:
            asks.append("git reset --hard discards uncommitted work.")
    elif sub == "clean":
        if "--force" in rest or short_flag(rest, "f", "e"):
            asks.append("git clean -f deletes untracked files.")
    elif sub in ("checkout", "switch", "restore"):
        takes = {"checkout": "bB", "switch": "cC", "restore": "s"}[sub]
        if (any(r in ("--force", "--discard-changes") for r in rest)
                or short_flag(rest, "f", takes)):
            asks.append(f"git {sub} --force discards uncommitted changes.")
        elif sub != "switch":
            pos = positionals(rest, takes, ("--source", "--pathspec-from-file"))
            staged_only = sub == "restore" and (
                ("--staged" in rest or short_flag(rest, "S", takes))
                and not ("--worktree" in rest or short_flag(rest, "W", takes)))
            if not staged_only and any(p in WIDE_PATHSPEC for p in pos):
                asks.append(f"git {sub} on the whole tree discards uncommitted "
                            "changes.")
    elif sub == "branch":
        forced = "--force" in rest or short_flag(rest, "f")
        deleting = "--delete" in rest or short_flag(rest, "d")
        if short_flag(rest, "D") or (forced and deleting):
            asks.append("Force-deleting a branch can lose unmerged commits.")
    elif sub == "stash":
        if rest and rest[0] in ("drop", "clear"):
            asks.append(f"git stash {rest[0]} deletes stashed work.")
    elif sub == "reflog":
        if rest and rest[0] in ("expire", "delete"):
            blocks.append("Expiring the reflog removes the way to recover work.")
    elif sub == "update-ref":
        if "-d" in rest:
            asks.append("git update-ref -d deletes a ref.")
    return blocks, asks


def analyze_rm(args, root, cwd):
    opts = [a for a in args if a.startswith("-") and a != "-"]
    recursive = any(a in ("-r", "-R", "--recursive")
                    or (not a.startswith("--") and ("r" in a or "R" in a))
                    for a in opts)
    targets = positionals(args)
    blocks, asks = [], []
    for t in targets:
        name = t.rstrip("/") or "/"
        rel = relative_to_root(t, root, cwd)
        if t in CATASTROPHIC or name in CATASTROPHIC or rel in ("", ".git"):
            blocks.append(f"rm {t} would delete the repo or your home directory.")
            continue
        if not recursive:
            continue
        base = os.path.basename(name)
        if base in DISPOSABLE or base.endswith(".egg-info"):
            continue
        if name.startswith(("/tmp/", "$TMPDIR", "${TMPDIR}")):
            continue
        asks.append(f"Recursive delete of {t}.")
    return blocks, asks


def allowed_secret_use(cmd, args, seg):
    if cmd in ("ls", "test", "[", "[[", "stat", "touch", "chmod", "cd"):
        return True
    if cmd == "git":
        _, sub_rest = strip_git_globals(args)
        if not sub_rest:
            return True
        sub, rest = sub_rest[0], sub_rest[1:]
        if sub in ("status", "check-ignore", "ls-files"):
            return True
        return sub == "rm" and "--cached" in rest
    if cmd in ("echo", "printf"):
        return bool(seg["out"]) and all(t.endswith(IGNORE_FILES) for t in seg["out"])
    if cmd in ("cp", "install"):
        pos = positionals(args)
        return not any(secret_file_mentions(p) for p in pos[:-1])
    return False


def strip_git_globals(args):
    i = 0
    while i < len(args) and args[i].startswith("-"):
        i += 2 if args[i] in GIT_GLOBAL_ARG else 1
    return args[:i], args[i:]


def write_targets(cmd, args):
    if cmd == "tee":
        return positionals(args)
    if cmd in ("sed", "perl") and any(a.startswith("-i") or a == "--in-place"
                                      or (a.startswith("-") and "i" in a and cmd == "perl")
                                      for a in args):
        return positionals(args, "ef")
    if cmd in ("cp", "install", "rsync", "ln"):
        pos = positionals(args, "t")
        return pos[-1:] if len(pos) > 1 else []
    if cmd in ("mv", "rm", "unlink", "truncate", "shred", "touch", "chmod", "chown"):
        return positionals(args, "s" if cmd == "truncate" else "")
    if cmd == "dd":
        return [a[3:] for a in args if a.startswith("of=")]
    return []


def analyze(command, root, cwd, depth=0):
    blocks, asks = [], []
    if depth > 3:
        return blocks, asks
    for seg in split_segments(tokenize(command)):
        raw = seg["words"]
        first = os.path.basename(raw[0]) if raw else ""
        if first in ("env", "printenv") and all(w.startswith("-") for w in raw[1:]):
            blocks.append(MSG_SECRET_PRINT)
            continue
        if first == "set" and len(raw) == 1:
            blocks.append(MSG_SECRET_PRINT)
            continue
        if first in ("export", "declare", "typeset"):
            if all(w.startswith("-") for w in raw[1:]):
                blocks.append(MSG_SECRET_PRINT)
                continue
            for w in raw[1:]:
                if HOOK_BYPASS_ENV.match(w):
                    blocks.append(MSG_HOOK_ENV.format(w.split("=")[0]))

        assigns, words = strip_wrappers(raw)
        for a in assigns:
            if HOOK_BYPASS_ENV.match(a) and words:
                blocks.append(MSG_HOOK_ENV.format(a.split("=")[0]))
        if not words:
            continue
        cmd, args = os.path.basename(words[0]), words[1:]

        if cmd == "cd" and len(args) == 1:
            cwd = str(Path(cwd, os.path.expanduser(args[0])))
            continue
        if cmd in SHELLS:
            for k, a in enumerate(args):
                if a.startswith("-") and not a.startswith("--") and a.endswith("c"):
                    if k + 1 < len(args):
                        b, s = analyze(args[k + 1], root, cwd, depth + 1)
                        blocks += b
                        asks += s
                    break
        if cmd == "eval":
            b, s = analyze(" ".join(args), root, cwd, depth + 1)
            blocks += b
            asks += s

        if cmd == "git":
            b, s = analyze_git(args)
            blocks += b
            asks += s
        if cmd == "rm":
            b, s = analyze_rm(args, root, cwd)
            blocks += b
            asks += s

        if cmd in ("echo", "printf") and any(SECRET_VAR.search(a) for a in args):
            blocks.append(MSG_SECRET_PRINT)
        if cmd == "printenv" and any(SECRET_VAR_NAME.match(a) for a in args):
            blocks.append(MSG_SECRET_PRINT)

        scan = [a for a in args
                if not a.startswith(("--exclude", "!", "--glob=!", "--iglob=!"))]
        mentioned = secret_file_mentions(" ".join(scan))
        read_via_input = secret_file_mentions(" ".join(seg["in"]))
        if read_via_input:
            blocks.append(MSG_SECRET_READ.format(read_via_input[0]))
        elif mentioned and not allowed_secret_use(cmd, args, seg):
            _, git_rest = strip_git_globals(args) if cmd == "git" else ([], [])
            if cmd == "git" and git_rest and git_rest[0] == "add":
                blocks.append(MSG_SECRET_ADD.format(mentioned[0]))
            else:
                blocks.append(MSG_SECRET_READ.format(mentioned[0]))

        for target in seg["out"] + write_targets(cmd, args):
            rel = relative_to_root(target, root, cwd)
            if rel is None:
                continue
            reason = protected_reason(rel, root) or protected_reason(
                rel.rstrip("/") + "/x", root)
            if reason and rel not in ("", "."):
                asks.append(f"{rel} is protected ({reason}).")
    return blocks, asks


def main():
    payload = read_payload()
    command = (payload.get("tool_input") or {}).get("command", "")
    if not command:
        return 0
    root = project_root(payload)
    cwd = payload.get("cwd") or str(root)
    blocks, asks = analyze(strip_text_heredocs(command), root, cwd)
    if blocks:
        print("Blocked: " + " ".join(dict.fromkeys(blocks)), file=sys.stderr)
        return 2
    if asks:
        print(json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "ask",
                "permissionDecisionReason": " ".join(dict.fromkeys(asks)),
            }
        }))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # a crash must not silently allow or brick Bash
        print(f"guard_bash.py failed ({exc!r}). The permission rules still apply.",
              file=sys.stderr)
        sys.exit(1)
