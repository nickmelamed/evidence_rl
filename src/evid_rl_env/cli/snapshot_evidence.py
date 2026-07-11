"""
CLI entry point: evid-snapshot

Pre-fetches and exports a portable JSON snapshot of Tavily evidence for a
dataset, so a later evid-train/evid-eval run can reproduce the exact same
evidence pool via --evidence-snapshot instead of depending on live Tavily
results (which change over time) or a machine-local sqlite cache.
"""

import argparse
import json

from evid_rl_env.data.dataset import load_dataset
from evid_rl_env.data.evidence_fetcher import export_snapshot, warm_cache


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a portable evidence snapshot for reproducible train/eval runs."
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Path to a JSON dataset file. Defaults to the full seed_claims.json.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/evidence_snapshot.json",
        help="Output JSON path.",
    )
    args = parser.parse_args()

    if args.dataset:
        with open(args.dataset) as f:
            dataset = json.load(f)
    else:
        dataset = load_dataset()

    warm_cache(dataset)
    export_snapshot(dataset, args.output)


if __name__ == "__main__":
    main()
