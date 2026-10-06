#!/usr/bin/env python3
"""Build the committed results tables from artifacts/experiments.

Every row names the run folder it came from, so a number in a document can be
traced back to a real run. Runs that lack a file are skipped for that table,
not filled in. Nothing is rounded beyond four decimals.

Example::

    python3 scripts/build_results_tables.py
"""
import argparse
import csv
import json
from pathlib import Path

TAIL = 50


def judge_architecture(cfg: dict) -> str:
    if cfg.get("judge_ensemble_models") and cfg.get("judge_escalation"):
        return f"escalating_{cfg.get('judge_escalation_target', 'ensemble')}"
    if cfg.get("judge_ensemble_models"):
        return "ensemble"
    return "single"


def load_config(run: Path) -> dict:
    path = run / "config.json"
    return json.loads(path.read_text()) if path.exists() else {}


def training_rows(runs: list[Path]) -> list[dict]:
    rows = []
    for run in runs:
        metrics = run / "metrics.csv"
        if not metrics.exists():
            continue
        with metrics.open() as f:
            data = list(csv.DictReader(f))
        rewards = [float(r["reward"]) for r in data if r.get("reward")]
        if not rewards:
            continue
        tail = rewards[-TAIL:]
        cfg = load_config(run)
        rows.append({
            "run": run.name,
            "algo": cfg.get("algo", ""),
            "judge_architecture": judge_architecture(cfg),
            "seed": cfg.get("seed", ""),
            "episodes_logged": len(rewards),
            "mean_reward_all": round(sum(rewards) / len(rewards), 4),
            f"mean_reward_last_{TAIL}": round(sum(tail) / len(tail), 4),
        })
    return rows


def r4(value):
    return round(value, 4) if isinstance(value, float) else value


def gold_rows(runs: list[Path]) -> list[dict]:
    rows = []
    for run in runs:
        path = run / "gold_eval.jsonl"
        if not path.exists():
            continue
        rounds = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not rounds:
            continue
        last = rounds[-1]
        cfg = load_config(run)
        rows.append({
            "run": run.name,
            "algo": cfg.get("algo", ""),
            "judge_architecture": judge_architecture(cfg),
            "rounds": len(rounds),
            "last_round_episode": last.get("episode", ""),
            "n_episodes": last.get("n_episodes", ""),
            "n_scored": last.get("n_scored", ""),
            "proxy_reward_mean": r4(last.get("proxy_reward_mean", "")),
            "gold_reward_mean": r4(last.get("gold_reward_mean", "")),
            "proxy_gold_correlation": r4(last.get("proxy_gold_correlation", "")),
            "outcome_accuracy_hard": r4(last.get("outcome_accuracy_hard", "")),
            "escalation_rate": r4(last.get("escalation_rate", "")),
        })
    return rows


def eval_rows(runs: list[Path]) -> list[dict]:
    rows = []
    for run in runs:
        path = run / "eval_results.json"
        if not path.exists():
            continue
        result = json.loads(path.read_text())
        for name, stats in result.get("baselines", {}).items():
            rows.append({
                "run": run.name,
                "n_episodes": result.get("n_episodes", ""),
                "policy": name,
                "reward_mean": stats.get("mean", ""),
                "reward_std": stats.get("std", ""),
                "rl_win_rate_vs_greedy": result.get("rl_win_rate_vs_greedy", ""),
            })
    return rows


def write(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--experiments", default="artifacts/experiments")
    parser.add_argument("--out", default="results/tables")
    args = parser.parse_args()

    runs = sorted(p for p in Path(args.experiments).iterdir() if p.is_dir())
    out = Path(args.out)
    write(out / "training_summary.csv", training_rows(runs))
    write(out / "gold_eval_last_round.csv", gold_rows(runs))
    write(out / "eval_baselines.csv", eval_rows(runs))


if __name__ == "__main__":
    main()
