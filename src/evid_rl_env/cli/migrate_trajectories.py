"""
evid-migrate: one-time migration of legacy JSONL trajectory records.

Scans data/ and logs/ for .jsonl files.  For each record that is missing any
of the three provenance fields introduced after the initial schema:

    collected_at   — ISO-8601 timestamp of when the trajectory was collected
    annotator_model — model or strategy that produced the actions
    mode           — collection mode (llm_annotator | reward_filtered | best_rollouts)

…it stamps default values derived from the file's modification time and writes
the patched records back to the same file in-place.

Use --dry-run to preview changes without writing anything to disk.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
from datetime import datetime, timezone

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

_SCAN_DIRS = ["data", "logs"]
_REQUIRED_FIELDS = {"collected_at", "annotator_model", "mode"}


def _mtime_iso(path: str) -> str:
    """Return the file's mtime as an ISO-8601 UTC string."""
    mtime = os.path.getmtime(path)
    return datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()


def _migrate_file(path: str, dry_run: bool) -> tuple[int, int]:
    """
    Process one .jsonl file.

    Returns:
        (migrated, skipped) — count of records patched vs already complete.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            raw_lines = fh.readlines()
    except OSError as exc:
        logger.warning("Cannot read %s: %s — skipped", path, exc)
        return 0, 0

    mtime_iso = _mtime_iso(path)
    patched_lines: list[str] = []
    migrated = 0
    skipped = 0
    changed = False

    for raw in raw_lines:
        stripped = raw.strip()
        if not stripped:
            patched_lines.append(raw)
            continue

        try:
            rec = json.loads(stripped)
        except json.JSONDecodeError:
            patched_lines.append(raw)
            continue

        if not isinstance(rec, dict):
            patched_lines.append(raw)
            continue

        missing = _REQUIRED_FIELDS - set(rec.keys())
        if missing:
            if dry_run:
                logger.info(
                    "[dry-run] Would patch record in %s | missing=%s | claim='%.60s'",
                    path,
                    sorted(missing),
                    rec.get("claim", "<no claim>"),
                )
            else:
                # Stamp sensible defaults; preserve any fields that are already set
                if "mode" not in rec:
                    rec["mode"] = "unknown"
                if "annotator_model" not in rec:
                    rec["annotator_model"] = "unknown"
                if "collected_at" not in rec:
                    rec["collected_at"] = mtime_iso
                patched_lines.append(json.dumps(rec, ensure_ascii=False) + "\n")
                changed = True
            migrated += 1
        else:
            patched_lines.append(raw)
            skipped += 1

    if changed and not dry_run:
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.writelines(patched_lines)
        except OSError as exc:
            logger.error("Failed to write %s: %s", path, exc)

    return migrated, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "One-time migration: stamp legacy JSONL trajectory records that are "
            "missing collected_at, annotator_model, or mode fields with safe defaults "
            "derived from file metadata."
        )
    )
    parser.add_argument(
        "--dirs",
        nargs="+",
        default=_SCAN_DIRS,
        metavar="DIR",
        help=f"Directories to scan recursively for .jsonl files (default: {_SCAN_DIRS}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would change without writing anything to disk.",
    )
    args = parser.parse_args()

    # Collect all .jsonl paths, deduplicating across recursive and flat globs
    all_files: list[str] = []
    for d in args.dirs:
        all_files.extend(glob.glob(os.path.join(d, "**", "*.jsonl"), recursive=True))
        all_files.extend(glob.glob(os.path.join(d, "*.jsonl")))
    all_files = sorted(set(all_files))

    if not all_files:
        print(f"No .jsonl files found in {args.dirs}")
        return

    total_migrated = 0
    total_skipped = 0
    files_touched = 0

    for path in all_files:
        m, s = _migrate_file(path, dry_run=args.dry_run)
        total_migrated += m
        total_skipped += s
        if m > 0:
            files_touched += 1
            action = "Would patch" if args.dry_run else "Patched"
            logger.info("%s: %s %d record(s), %d already complete", path, action, m, s)

    prefix = "[dry-run] " if args.dry_run else ""
    print(
        f"{prefix}Migrated {total_migrated} records across {files_touched} files "
        f"| Skipped {total_skipped} already-complete records"
    )


if __name__ == "__main__":
    main()
