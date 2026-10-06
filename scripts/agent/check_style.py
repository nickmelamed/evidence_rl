#!/usr/bin/env python3
"""Flag comment, docstring, and Markdown patterns that read as machine-written.

Usage::

    check_style.py [PATHS...]            # whole files
    check_style.py --changed [PATHS...]  # only lines changed on this branch
    check_style.py --hook                # PostToolUse hook, edited file only

Checks Python comments and docstrings, R comments, Markdown and Quarto prose,
and notebook cells. ``--changed`` compares against the merge base with the
default branch plus uncommitted and untracked files, so legacy text outside
the diff never blocks work. The hook uses the same rule for the edited file.

Add ``style: ok`` to a line only when its text is quoted verbatim from an
outside source. Globs in ``.claude/style-ignore`` exclude files.
``.claude/style-words`` adjusts the word list (``+word`` adds, ``-word``
removes).
"""
import argparse
import ast
import fnmatch
import io
import json
import os
import re
import sys
import tokenize
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import base_ref, git, has_head, project_root, read_payload  # noqa: E402

SKIP_DIRS = {
    ".git", ".venv", "venv", "env", "node_modules", "__pycache__", "build",
    "dist", ".mypy_cache", ".pytest_cache", ".ruff_cache", "renv", ".quarto",
    ".tox", ".nox", ".ipynb_checkpoints", "site-packages", "_site", "data",
}
DEFAULT_IGNORE = [
    "CLAUDE.md", "AGENTS.md", "PROGRESS.md", "CHANGELOG.md", ".claude/*",
    "*handoff*.md", "vendor/*", "third_party/*", "LICENSE*",
]
IGNORE_MARK = "style: ok"
EXTENSIONS = {".py", ".md", ".qmd", ".rmd", ".r", ".ipynb"}

# These are the words writing-style.md lists. The test suite checks the two
# stay in sync.
DEFAULT_WORDS = [
    "seamless", "seamlessly", "comprehensive", "leverage", "leverages",
    "leveraged", "leveraging", "delve", "delves", "delving", "crucial",
    "utilize", "utilizes", "utilized", "utilizing", "meticulous",
    "meticulously",
]

# An en dash between two digits is a numeric range ("images 21–40").
DASH = re.compile(r"—|(?<!\d)–|–(?!\d)")
AGENT_REF = re.compile(r"CLAUDE\.md|AGENTS\.md|\bas instructed\b", re.I)
HISTORY = re.compile(
    r"^(now (handles|supports|uses|returns|accepts|works|also)\b"
    r"|(updated|changed|modified) to\b"
    r"|refactored\b"
    r"|fixed (a |an |the )?(bug|issue|crash|typo|regression)\b"
    r"|(added|removed) (this|the) (check|line|function|code)\b)",
    re.I,
)
PRAGMA = re.compile(
    r"^(!|type:|noqa|pragma|pylint:|mypy:|pyright:|fmt:|isort:|ruff:|nosec|"
    r"vim:|coding[:=]|nolint|region\b|endregion\b)|-\*-.*-\*-",
    re.I,
)
BANNER = re.compile(r"^[=\-*#~_/+ ]{3,}$|^[=\-*#~_]{3,}\s*\S.*\S\s*[=\-*#~_]{3,}$")
BOLD_LEADIN = re.compile(r"\*\*[^*]+:\*\*|\*\*[^*]+\*\*\s*:")
INLINE_CODE = re.compile(r"`+[^`]*`+")
LINK_TARGET = re.compile(r"\]\([^)]*\)|<https?://[^>]*>|https?://\S+")
HTML_BITS = re.compile(r"&#?\w+;|<[^>]+>")
FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")


def word_pattern(root):
    words = set(DEFAULT_WORDS)
    f = root / ".claude" / "style-words"
    if f.is_file():
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("-"):
                words.discard(line[1:].strip().lower())
            else:
                words.add(line.lstrip("+").strip().lower())
    if not words:
        return None
    alts = "|".join(sorted(map(re.escape, words), key=len, reverse=True))
    return re.compile(rf"\b({alts})\b", re.I)


def clean_prose(text):
    return HTML_BITS.sub("", LINK_TARGET.sub("", INLINE_CODE.sub("", text)))


def prose_problems(text, words, allow_semicolon=False, in_code=True):
    text = clean_prose(text)
    found = []
    if DASH.search(text):
        found.append("em or en dash used as punctuation")
    m = words.search(text) if words else None
    if m:
        found.append(f"filler word '{m.group(0)}'")
    if in_code and AGENT_REF.search(text):
        found.append("references agent instructions")
    if not allow_semicolon and ";" in text:
        found.append("semicolon in prose")
    return found


def comment_problems(body, words, allow_semicolon=True):
    if not body or PRAGMA.search(body):
        return []
    probs = prose_problems(body, words, allow_semicolon)
    if HISTORY.search(body):
        probs.append("describes edit history")
    if BANNER.search(body):
        probs.append("section banner")
    return probs


def check_python(source, words):
    findings = []
    lines = source.splitlines()
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type != tokenize.COMMENT:
                continue
            lineno = tok.start[0]
            if IGNORE_MARK in lines[lineno - 1]:
                continue
            body = tok.string.lstrip("#").strip()
            findings += [(lineno, p) for p in comment_problems(body, words)]
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return findings
    nodes = [tree] + [n for n in ast.walk(tree) if isinstance(
        n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    for node in nodes:
        body = getattr(node, "body", [])
        if not body or not isinstance(body[0], ast.Expr):
            continue
        val = body[0].value
        if not (isinstance(val, ast.Constant) and isinstance(val.value, str)):
            continue
        in_doctest = False
        for offset, line in enumerate(val.value.splitlines()):
            lineno = val.lineno + offset
            stripped = line.strip()
            if stripped.startswith(">>>"):
                in_doctest = True
            elif not stripped:
                in_doctest = False
            if in_doctest or (lineno <= len(lines) and IGNORE_MARK in lines[lineno - 1]):
                continue
            findings += [(lineno, p) for p in prose_problems(line, words)]
    return findings


def r_comment(line):
    """Return the comment text of an R line, or None."""
    quote = None
    for i, ch in enumerate(line):
        if quote:
            if ch == "\\":
                continue
            if ch == quote and (i == 0 or line[i - 1] != "\\"):
                quote = None
        elif ch in "\"'`":
            quote = ch
        elif ch == "#":
            return line[i:]
    return None


def check_r(source, words):
    findings = []
    for lineno, line in enumerate(source.splitlines(), 1):
        comment = r_comment(line)
        if comment is None or IGNORE_MARK in line:
            continue
        roxygen = comment.startswith("#'")
        body = comment.lstrip("#'").strip()
        findings += [(lineno, p) for p in
                     comment_problems(body, words, allow_semicolon=not roxygen)]
    return findings


def check_markdown(source, words):
    findings = []
    lines = source.splitlines()
    fence = None
    in_html_comment = False
    start = 0
    if lines and lines[0].strip() == "---":
        for k in range(1, len(lines)):
            if lines[k].strip() in ("---", "..."):
                start = k + 1
                break
    prev_blank = True
    for lineno, line in enumerate(lines[start:], start + 1):
        stripped = line.strip()
        m = FENCE.match(line)
        if fence:
            if m and m.group(1)[0] == fence[0] and len(m.group(1)) >= len(fence) \
                    and not line[m.end():].strip():
                fence = None
            continue
        if m:
            fence = m.group(1)
            continue
        if in_html_comment or stripped.startswith("<!--"):
            in_html_comment = "-->" not in stripped
            continue
        indented_code = prev_blank and (line.startswith("    ") or line.startswith("\t")) \
            and not stripped.startswith(("-", "*", "+")) and not re.match(r"\d+\.", stripped)
        prev_blank = not stripped
        if indented_code or IGNORE_MARK in line:
            continue
        probs = prose_problems(line, words, in_code=False)
        if BOLD_LEADIN.search(INLINE_CODE.sub("", line)):
            probs.append("bold lead-in with colon")
        findings += [(lineno, p) for p in probs]
    return findings


def check_notebook(source, words):
    try:
        nb = json.loads(source)
    except json.JSONDecodeError:
        return []
    findings = []
    for n, cell in enumerate(nb.get("cells", []), 1):
        text = "".join(cell.get("source", []))
        kind = cell.get("cell_type")
        if kind == "markdown":
            found = check_markdown(text, words)
        elif kind == "code":
            found = check_python(text, words)
        else:
            continue
        findings += [(f"cell {n} line {ln}", p) for ln, p in found]
    return findings


def check_source(path, source, words):
    ext = path.suffix.lower()
    if ext == ".py":
        return check_python(source, words)
    if ext == ".r":
        return check_r(source, words)
    if ext in (".md", ".qmd", ".rmd"):
        return check_markdown(source, words)
    if ext == ".ipynb":
        return check_notebook(source, words)
    return []


def load_ignore(root):
    patterns = list(DEFAULT_IGNORE)
    f = root / ".claude" / "style-ignore"
    if f.is_file():
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                patterns.append(line)
    return patterns


def rel_path(path, root):
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return None


def is_ignored(rel, patterns):
    name = rel.rsplit("/", 1)[-1]
    return any(fnmatch.fnmatchcase(rel, p) or fnmatch.fnmatchcase(name, p)
               for p in patterns)


def list_files(paths, root):
    """Yield candidate files, using git so ignored and vendored trees are skipped."""
    for raw in paths:
        p = Path(raw)
        if p.is_file():
            yield p
            continue
        if not p.is_dir():
            continue
        listed = git(root, "ls-files", "-co", "--exclude-standard", "--",
                     str(p.resolve()))
        if listed or git(root, "rev-parse", "--is-inside-work-tree").strip():
            for line in listed.splitlines():
                f = root / line
                if f.is_file():
                    yield f
            continue
        for dirpath, dirnames, filenames in os.walk(p):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for name in filenames:
                yield Path(dirpath) / name


def diff_base(root):
    base = base_ref(root)
    if base:
        return base
    return "HEAD" if has_head(root) else None


def changed_lines(root, rel, base):
    """Return the set of changed line numbers, or None for "every line"."""
    if base is None:
        return None
    if not git(root, "ls-files", "--", rel).strip():
        return None
    if git(root, "cat-file", "-t", f"{base}:{rel}").strip() != "blob":
        return None
    out = git(root, "diff", "-U0", "--no-color", "--no-ext-diff", base, "--", rel)
    lines = set()
    for m in re.finditer(r"^@@ -\S+ \+(\d+)(?:,(\d+))? @@", out, re.M):
        start, count = int(m.group(1)), int(m.group(2) or 1)
        lines.update(range(start, start + count))
    return lines


def changed_files(root, base):
    names = set(git(root, "ls-files", "--others", "--exclude-standard").splitlines())
    if base:
        names |= set(git(root, "diff", "--name-only", "--diff-filter=ACMR",
                         base).splitlines())
    return names


def excerpt(source, where):
    if not isinstance(where, int):
        return ""
    lines = source.splitlines()
    if not 0 < where <= len(lines):
        return ""
    text = lines[where - 1].strip()
    return f'  "{text[:70]}{"..." if len(text) > 70 else ""}"'


def run(files, root, changed_only, words, patterns):
    if changed_only and not git(root, "rev-parse", "--is-inside-work-tree").strip():
        changed_only = False
    base = diff_base(root) if changed_only else None
    wanted = changed_files(root, base) if changed_only else None
    report = []
    for f in files:
        rel = rel_path(f, root)
        if rel is None or f.suffix.lower() not in EXTENSIONS:
            continue
        if is_ignored(rel, patterns) or (wanted is not None and rel not in wanted):
            continue
        try:
            source = f.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        findings = check_source(f, source, words)
        if changed_only and f.suffix.lower() != ".ipynb":
            keep = changed_lines(root, rel, base)
            if keep is not None:
                findings = [(ln, p) for ln, p in findings if ln in keep]
        for where, problem in findings:
            report.append(f"{rel}:{where}: {problem}{excerpt(source, where)}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", default=["."])
    parser.add_argument("--changed", action="store_true",
                        help="only check lines changed on this branch")
    parser.add_argument("--hook", action="store_true",
                        help="read a Claude Code hook payload from stdin")
    args = parser.parse_args()

    if args.hook:
        payload = read_payload()
        tool_input = payload.get("tool_input") or {}
        target = tool_input.get("file_path") or tool_input.get("notebook_path")
        if not target:
            return 0
        root = project_root(payload)
        path = Path(target)
        if not path.is_absolute():
            path = Path(payload.get("cwd") or root) / path
        files, changed_only = [path], True
    else:
        root = project_root()
        files, changed_only = list_files(args.paths, root), args.changed

    report = run(files, root, changed_only, word_pattern(root), load_ignore(root))
    if not report:
        return 0
    out = sys.stderr if args.hook else sys.stdout
    print("\n".join(report), file=out)
    if args.hook:
        print("Fix these in the lines you wrote before continuing. The rules are "
              "in .claude/rules/writing-style.md.", file=out)
        return 2
    return 1


if __name__ == "__main__":
    sys.exit(main())
