#!/usr/bin/env python3
"""Check that every reported number in the docs comes from a generated artifact.

Finds decimals, percentages, and scientific notation in the given documents
(outside code, links, front matter, and HTML comments) and fails if a number
does not match any number in the source files. A reported number matches
when a source value rounds to it at the precision it was written with, so
``87.3%`` matches a source value of ``0.8734`` or ``87.34``, and ``0.873``
matches ``0.8734``. Plain integers are not checked.

Version numbers (``Python 3.11``, ``v1.2``, ``numpy>=1.26``) and thresholds
(``p < 0.05``) are skipped. Write ``TBD`` for numbers that do not exist yet,
and add ``numbers: ok`` to a line whose numbers are not results.

Example::

    python3 scripts/agent/check_numbers.py README.md MODEL_CARD.md \\
        --sources results/tables results/release
"""
import argparse
import bisect
import re
import sys
from pathlib import Path

IGNORE_MARK = "numbers: ok"
REPORTED = re.compile(
    r"(?<![\w.])(\d{1,3}(?:,\d{3})+\.\d+%?|\d{1,3}(?:,\d{3})+%"
    r"|\d+\.\d+(?:[eE][-+]?\d+)?%?|\d+[eE][-+]?\d+%?|\d+%)(?![\w.]*\d)"
)
SOURCE = re.compile(r"(?<![\w.])-?(\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)(%?)")
LINK_TARGET = re.compile(r"\]\([^)]*\)|<https?://[^>]*>|https?://\S+")
INLINE_CODE = re.compile(r"`+[^`]*`+")
VERSION = re.compile(
    r"(?i)(\b(?:python|py|r|cuda|cudnn|node|nodejs|java|go|rust|ubuntu|debian|"
    r"macos|windows|ios|android|numpy|pandas|torch|pytorch|tensorflow|jax|"
    r"scikit-learn|sklearn|version|release)\s+v?\d+(?:\.\d+)+\b)"
    r"|\bv\d+(?:\.\d+)+\b|\b\d+\.\d+\.\d+(?:\.\d+)*\b"
    r"|[\w\]](?:==|>=|<=|~=|!=|\^)\s*\d+(?:\.\d+)+"
)
THRESHOLD = re.compile(r"(?i)\b(?:p|alpha|α)\s*(?:[<>≤≥]=?|=)\s*0?\.\d+")
FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")


def reported_numbers(text):
    lines = text.splitlines()
    start = 0
    if lines and lines[0].strip() == "---":
        for k in range(1, len(lines)):
            if lines[k].strip() in ("---", "..."):
                start = k + 1
                break
    fence, in_comment = None, False
    for lineno, line in enumerate(lines[start:], start + 1):
        stripped = line.strip()
        m = FENCE.match(line)
        if fence:
            if m and m.group(1)[0] == fence[0] and len(m.group(1)) >= len(fence):
                fence = None
            continue
        if m:
            fence = m.group(1)
            continue
        if in_comment or stripped.startswith("<!--"):
            in_comment = "-->" not in stripped
            continue
        if IGNORE_MARK in line:
            continue
        clean = INLINE_CODE.sub(" ", LINK_TARGET.sub(" ", line))
        clean = THRESHOLD.sub(" ", VERSION.sub(" ", clean))
        for m in REPORTED.finditer(clean):
            yield lineno, m.group(1)


def parse_reported(token):
    """Return (value, half_ulp, is_percent) for a number as written."""
    is_percent = token.endswith("%")
    body = token.rstrip("%").replace(",", "")
    mantissa, _, exp = body.lower().partition("e")
    decimals = len(mantissa.split(".")[1]) if "." in mantissa else 0
    half = 0.5 * 10 ** (-decimals) * (10 ** int(exp) if exp else 1)
    return float(body), half, is_percent


def source_values(dirs):
    values = set()
    for d in dirs:
        p = Path(d)
        if not p.exists():
            continue
        files = [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file()]
        for f in files:
            try:
                text = f.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for num, pct in SOURCE.findall(text):
                try:
                    v = float(num)
                except ValueError:
                    continue
                values.add(v)
                if pct:
                    values.add(v / 100)
    return sorted(values)


def has_value_near(values, target, half):
    tol = half * (1 + 1e-9) + 1e-12
    i = bisect.bisect_left(values, target - tol)
    return i < len(values) and values[i] <= target + tol


def matches(values, token):
    value, half, is_percent = parse_reported(token)
    if has_value_near(values, value, half):
        return True
    return is_percent and has_value_near(values, value / 100, half / 100)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("docs", nargs="+")
    parser.add_argument("--sources", nargs="+", default=["results/tables"])
    parser.add_argument("--require-sources", action="store_true",
                        help="fail instead of skipping when no source exists")
    args = parser.parse_args()

    missing = [d for d in args.docs if not Path(d).is_file()]
    if missing:
        print(f"check_numbers: document not found: {', '.join(missing)}",
              file=sys.stderr)
        return 2
    if not any(Path(s).exists() for s in args.sources):
        msg = f"check_numbers: no source found in {', '.join(args.sources)}"
        if args.require_sources:
            print(msg, file=sys.stderr)
            return 2
        print(msg + ", skipping until results exist.", file=sys.stderr)
        return 0
    values = source_values(args.sources)

    problems = []
    for doc in args.docs:
        text = Path(doc).read_text(encoding="utf-8")
        for lineno, token in reported_numbers(text):
            if not matches(values, token):
                problems.append(f"{doc}:{lineno}: {token} is not in "
                                f"{', '.join(args.sources)}")
    if problems:
        print("\n".join(problems))
        print("Regenerate the tables and copy numbers from them, write TBD, "
              "or mark a non-result line with 'numbers: ok'.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
