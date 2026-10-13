"""
Evaluate a trained debate pair on the held-out split.

Run with ``python -m evid_rl_env.cli.eval_debate --run-dir artifacts/experiments/<run>``.
The run folder holds ``policy_support.npz`` and ``policy_contradict.npz``. The
pair is scored on the same single-judge env and the same claims as ``evid-eval``,
so the number is comparable with the single-agent runs. Results go to
``eval_results.json`` in the run folder.
"""

from __future__ import annotations

import argparse
import json
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import random
from pathlib import Path

import numpy as np

from evid_rl_env.agent.config_loader import load_base_config
from evid_rl_env.agent.debate_evaluator import DebateEvaluator
from evid_rl_env.agent.policy import ActorCriticPolicy
from evid_rl_env.data.dataset import load_dataset, split_dataset
from evid_rl_env.data.evidence_fetcher import use_snapshot
from evid_rl_env.environment.environment import ClaimEnv


def evaluate_run(run_dir: str, eval_dataset: list, n_episodes: int, seed: int,
                 turns_per_side: int | None = None) -> dict:
    """Score the pair saved in ``run_dir`` and write ``eval_results.json`` there."""
    run = Path(run_dir)
    support_path = run / "policy_support.npz"
    contradict_path = run / "policy_contradict.npz"
    for path in (support_path, contradict_path):
        if not path.exists():
            raise FileNotFoundError(f"missing checkpoint: {path}")

    if turns_per_side is None:
        config_path = run / "config.json"
        turns_per_side = 5
        if config_path.exists():
            turns_per_side = json.loads(config_path.read_text()).get("turns_per_side", 5)

    support = ActorCriticPolicy.load(str(support_path), seed=seed)
    # One loaded actor model writes the text for both roles.
    contradict = ActorCriticPolicy.load(str(contradict_path), seed=seed, llm=support.llm)

    random.seed(seed)
    np.random.seed(seed)
    env = ClaimEnv(eval_dataset, seed=seed)
    evaluator = DebateEvaluator(
        env, support, contradict, turns_per_side=turns_per_side, n_eval_episodes=n_episodes
    )
    result = evaluator.evaluate()

    out = {
        "checkpoint": str(run),
        "n_episodes": n_episodes,
        "evaluator": "debate_pair",
        "baselines": {"rl": {
            "mean": round(result["eval/mean_reward_raw"], 4),
            "std": round(result["eval/std_reward_raw"], 4),
        }},
        "judge_metrics": {},
    }
    (run / "eval_results.json").write_text(json.dumps(out, indent=2))
    print(f"Eval results written to {run / 'eval_results.json'}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained debate pair.")
    parser.add_argument("--run-dir", required=True,
                        help="Run folder with policy_support.npz and policy_contradict.npz.")
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None,
                        help="Defaults to configs/base.yaml. Use the same seed as evid-eval "
                             "so every method sees the same episodes.")
    parser.add_argument("--evidence-snapshot", type=str, default=None)
    args = parser.parse_args()

    if args.evidence_snapshot is not None:
        use_snapshot(args.evidence_snapshot)
    seed = args.seed if args.seed is not None else load_base_config().get("seed", 42)

    _, eval_dataset = split_dataset(load_dataset())
    result = evaluate_run(args.run_dir, eval_dataset, args.n_episodes, seed)
    print(f"Debate pair: {result['eval/mean_reward_raw']:.3f} ± {result['eval/std_reward_raw']:.3f}")


if __name__ == "__main__":
    main()
