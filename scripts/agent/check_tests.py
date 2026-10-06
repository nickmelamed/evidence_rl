#!/usr/bin/env python3
"""Detect tests that were weakened compared with the base branch.

Compares each changed test file (pytest, unittest, or testthat) with its
version at the merge base and reports when

- assertions disappear, or become trivial (``assert True``),
- specific assertions (``assert x == 5``) turn into bare ones (``assert x``),
- skip or xfail markers appear,
- a tolerance loosens (``rel``, ``abs``, ``atol``, ``rtol``, ``tolerance``,
  or fewer ``decimal``/``places``),
- conftest.py or the pytest configuration starts skipping or deselecting.

Renamed files are compared with their old path. This is a heuristic next to
the spec-reviewer agent, not a proof that tests are unchanged in meaning.

The Stop hook calls this. CI runs it with ``--warn`` so a change the owner
approved in the PR does not fail the build. ``AGENT_ALLOW_TEST_CHANGES=1`` in
the environment that launches Claude Code turns the check off for a session.
"""
import argparse
import ast
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import base_ref, git, has_head, project_root  # noqa: E402

PY_SKIPS = re.compile(
    r"pytest\.mark\.(skip|skipif|xfail)\b|pytest\.(skip|xfail)\(|"
    r"unittest\.(skip|skipIf|skipUnless|expectedFailure)\b|\.skipTest\(|@skip\b"
)
R_ASSERTS = re.compile(r"\b(expect_\w+|stopifnot)\s*\(")
R_TRIVIAL = re.compile(r"\bexpect_true\s*\(\s*TRUE\s*\)")
R_SKIPS = re.compile(r"\bskip(_if\w*|_on_\w+)?\s*\(")
R_TOL = re.compile(r"\btolerance\s*=\s*([0-9.eE+-]+)")
TOL_LOOSER_WHEN_LARGER = {"rel", "abs", "atol", "rtol", "delta", "rel_tol",
                          "abs_tol", "tolerance"}
TOL_LOOSER_WHEN_SMALLER = {"decimal", "places"}
CONFIG_FILES = {"pyproject.toml", "pytest.ini", "setup.cfg", "tox.ini"}
DESELECT = re.compile(r"--deselect|--ignore|(^|\s)-k\s|-m\s*[\"']?not\b|"
                      r"collect_ignore|\bskip\b|\bxfail\b")


def call_name(func):
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


COMPARING_CALLS = {
    "isinstance", "issubclass", "allclose", "isclose", "array_equal",
    "array_equiv", "equals", "equal", "match", "fullmatch", "search",
    "startswith", "endswith", "issubset", "issuperset",
}


def is_specific(test):
    """Return True if an assert compares against something, not just truthiness."""
    if isinstance(test, ast.Compare):
        return True
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        return is_specific(test.operand)
    if isinstance(test, ast.BoolOp):
        return all(is_specific(v) for v in test.values)
    return isinstance(test, ast.Call) and call_name(test.func) in COMPARING_CALLS


def number(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = number(node.operand)
        return -inner if inner is not None else None
    return None


def py_metrics(src):
    m = {"asserts": 0, "specific": 0, "trivial": 0,
         "skips": len(PY_SKIPS.findall(src)), "tol": {}}
    try:
        tree = ast.parse(src)
    except SyntaxError:
        m["asserts"] = len(re.findall(r"^\s*assert\b|\.assert\w*\(|assert_\w+\(",
                                      src, re.M))
        return m
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            test = node.test
            if isinstance(test, ast.Constant):
                m["trivial"] += 1
                continue
            m["asserts"] += 1
            if is_specific(test):
                m["specific"] += 1
        elif isinstance(node, ast.Call):
            name = call_name(node.func)
            if name in ("raises", "warns", "deprecated_call") or (
                    name.startswith("assert") and name not in ("assertTrue",)):
                m["asserts"] += 1
                m["specific"] += 1
            elif name == "assertTrue":
                arg = node.args[0] if node.args else None
                if isinstance(arg, ast.Constant):
                    m["trivial"] += 1
                else:
                    m["asserts"] += 1
            for kw in node.keywords:
                value = number(kw.value)
                if kw.arg and value is not None and (
                        kw.arg in TOL_LOOSER_WHEN_LARGER
                        or kw.arg in TOL_LOOSER_WHEN_SMALLER):
                    m["tol"].setdefault(kw.arg, []).append(value)
    return m


def r_metrics(src):
    tol = [float(v) for v in R_TOL.findall(src) if _is_float(v)]
    trivial = len(R_TRIVIAL.findall(src))
    asserts = len(R_ASSERTS.findall(src)) - trivial
    return {"asserts": asserts, "specific": asserts, "trivial": trivial,
            "skips": len(R_SKIPS.findall(src)),
            "tol": {"tolerance": tol} if tol else {}}


def _is_float(text):
    try:
        float(text)
        return True
    except ValueError:
        return False


def is_test_file(path):
    name = path.rsplit("/", 1)[-1]
    lower = name.lower()
    in_tests = path.startswith("tests/") or "/tests/" in path
    if lower.endswith(".py") and name != "conftest.py":
        return in_tests or lower.startswith("test_") or lower.endswith("_test.py")
    if lower.endswith(".r"):
        return "testthat/" in path or lower.startswith(("test-", "test_"))
    return False


def compare(path, before, after):
    metrics = r_metrics if path.lower().endswith(".r") else py_metrics
    b, a = metrics(before), metrics(after)
    problems = []
    if a["asserts"] < b["asserts"]:
        problems.append(f"assertions went from {b['asserts']} to {a['asserts']}")
    elif a["specific"] < b["specific"]:
        problems.append(f"specific assertions went from {b['specific']} to "
                        f"{a['specific']} (a check became weaker)")
    if a["trivial"] > b["trivial"]:
        problems.append(f"trivial assertions went from {b['trivial']} to "
                        f"{a['trivial']}")
    if a["skips"] > b["skips"]:
        problems.append(f"skip/xfail markers went from {b['skips']} to {a['skips']}")
    for kw in set(a["tol"]) & set(b["tol"]):
        if kw in TOL_LOOSER_WHEN_SMALLER:
            old, new = min(b["tol"][kw]), min(a["tol"][kw])
            looser = new < old
        else:
            old, new = max(b["tol"][kw]), max(a["tol"][kw])
            looser = new > old
        if looser:
            problems.append(f"tolerance {kw} loosened from {old:g} to {new:g}")
    return [f"{path}: {p}" for p in problems]


def added_lines(root, base, path):
    diff = git(root, "diff", "-U0", "--no-color", base, "--", path)
    return [line[1:] for line in diff.splitlines()
            if line.startswith("+") and not line.startswith("+++")]


def weakened_tests(root, base=None):
    if os.environ.get("AGENT_ALLOW_TEST_CHANGES") == "1":
        return []
    base = base or base_ref(root) or ("HEAD" if has_head(root) else None)
    if base is None:
        return []
    problems = []
    status = git(root, "diff", "-M", "--name-status", base)
    for line in status.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        code, old = parts[0], parts[1]
        new = parts[2] if code.startswith(("R", "C")) and len(parts) > 2 else old
        name = new.rsplit("/", 1)[-1]
        if name == "conftest.py" or name in CONFIG_FILES:
            hits = [text.strip() for text in added_lines(root, base, new)
                    if DESELECT.search(text)]
            if hits:
                problems.append(f"{new}: new lines may skip or deselect tests: "
                                + "; ".join(hits[:3]))
            continue
        if not (is_test_file(old) or is_test_file(new)):
            continue
        if code.startswith("A"):
            continue
        before = git(root, "show", f"{base}:{old}")
        f = root / new
        after = "" if code.startswith("D") or not f.is_file() else \
            f.read_text(encoding="utf-8", errors="replace")
        problems += compare(new, before, after)
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", help="compare with this ref instead of the "
                                       "merge base with the default branch")
    parser.add_argument("--warn", action="store_true",
                        help="print problems but exit 0 (for CI)")
    args = parser.parse_args()
    problems = weakened_tests(project_root(), args.base)
    if not problems:
        return 0
    label = "Warning, tests" if args.warn else "Tests"
    print(f"{label} weakened compared with the base branch:")
    print("\n".join(problems))
    return 0 if args.warn else 1


if __name__ == "__main__":
    sys.exit(main())
