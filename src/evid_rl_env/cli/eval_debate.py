"""
Evaluate a trained debate pair on the held-out split.

Run with ``python -m evid_rl_env.cli.eval_debate --run-dir artifacts/experiments/<run>``.
The run folder holds ``policy_support.npz`` and ``policy_contradict.npz``. The
pair is scored on the same single-judge env and the same claims as ``evid-eval``,
so the number is comparable with the single-agent runs. Results go to
``eval_results.json`` in the run folder.

With ``--gold-judge-model`` it instead audits the run's arbiter against that
held-out judge and appends a row to ``gold_eval.jsonl``.
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
from evid_rl_env.agent.debate_gold import build_debate_gold_evaluator
from evid_rl_env.agent.llm_client import JudgeLLMClient
from evid_rl_env.agent.policy import ActorCriticPolicy
from evid_rl_env.data.dataset import load_dataset, split_dataset
from evid_rl_env.data.evidence_fetcher import use_snapshot
from evid_rl_env.environment.environment import ClaimEnv
from evid_rl_env.judge.arbiter import build_arbiter
from evid_rl_env.judge.llm_judge import LLMJudge


def _run_config(run: Path) -> dict:
    config_path = run / "config.json"
    return json.loads(config_path.read_text()) if config_path.exists() else {}


def _load_pair(run: Path, seed: int):
    support_path = run / "policy_support.npz"
    contradict_path = run / "policy_contradict.npz"
    for path in (support_path, contradict_path):
        if not path.exists():
            raise FileNotFoundError(f"missing checkpoint: {path}")
    support = ActorCriticPolicy.load(str(support_path), seed=seed)
    # One loaded actor model writes the text for both roles.
    contradict = ActorCriticPolicy.load(str(contradict_path), seed=seed, llm=support.llm)
    return support, contradict


def evaluate_run(run_dir: str, eval_dataset: list, n_episodes: int, seed: int,
                 turns_per_side: int | None = None) -> dict:
    """Score the pair saved in ``run_dir`` and write ``eval_results.json`` there."""
    run = Path(run_dir)
    cfg = _run_config(run)
    support, contradict = _load_pair(run, seed)
    if turns_per_side is None:
        turns_per_side = cfg.get("turns_per_side", 5)

    random.seed(seed)
    np.random.seed(seed)
    env = ClaimEnv(eval_dataset, seed=seed)
    evaluator = DebateEvaluator(
        env, support, contradict, turns_per_side=turns_per_side, n_eval_episodes=n_episodes,
        max_tokens=cfg.get("max_tokens_per_episode"),
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


def gold_audit_run(run_dir: str, eval_dataset: list, gold_judge_model: str, n_episodes: int,
                   seed: int) -> dict:
    """Audit the run's arbiter against a held-out judge and append a row to ``gold_eval.jsonl``."""
    run = Path(run_dir)
    cfg = _run_config(run)
    arbiter_model = cfg.get("arbiter_model")
    if not arbiter_model:
        raise ValueError(f"{run / 'config.json'} does not record an arbiter_model")
    if arbiter_model == gold_judge_model:
        raise ValueError(f"the gold judge must differ from the arbiter, both are {arbiter_model}")
    support, contradict = _load_pair(run, seed)

    random.seed(seed)
    gold_subset = random.sample(eval_dataset, min(n_episodes, len(eval_dataset)))
    arbiter = build_arbiter(arbiter_model, seed)
    gold_env = ClaimEnv(
        gold_subset, judge_model=cfg.get("judge_model"), seed=seed, llm_judge=arbiter,
    )
    gold_judge = LLMJudge(
        JudgeLLMClient(model_name=gold_judge_model, seed=seed),
        cache_path="artifacts/cache/gold_judge_cache.sqlite3",
    )
    evaluator = build_debate_gold_evaluator(
        gold_env, support, contradict, [gold_judge], n_episodes=len(gold_subset),
        turns_per_side=cfg.get("turns_per_side", 5), max_tokens=cfg.get("max_tokens_per_episode"),
    )
    np.random.seed(seed)
    summary = evaluator.evaluate()

    row = {"episode": None, "checkpoint": str(run), "gold_judge_model": gold_judge_model,
           **summary}
    with open(run / "gold_eval.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"Gold audit appended to {run / 'gold_eval.jsonl'}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained debate pair.")
    parser.add_argument("--run-dir", required=True,
                        help="Run folder with policy_support.npz and policy_contradict.npz.")
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None,
                        help="Defaults to configs/base.yaml. Use the same seed as evid-eval "
                             "so every method sees the same episodes.")
    parser.add_argument("--gold-judge-model", default=None,
                        help="Audit the run's arbiter against this held-out judge instead of "
                             "scoring the pair. Must differ from the arbiter.")
    parser.add_argument("--evidence-snapshot", type=str, default=None)
    args = parser.parse_args()

    if args.evidence_snapshot is not None:
        use_snapshot(args.evidence_snapshot)
    seed = args.seed if args.seed is not None else load_base_config().get("seed", 42)

    _, eval_dataset = split_dataset(load_dataset())
    if args.gold_judge_model:
        summary = gold_audit_run(args.run_dir, eval_dataset, args.gold_judge_model,
                                 args.n_episodes, seed)
        print(f"Scored {summary['n_scored']}/{summary['n_episodes']} episodes | "
              f"proxy={summary['proxy_reward_mean']} gold={summary['gold_reward_mean']} "
              f"corr={summary['proxy_gold_correlation']}")
        return
    result = evaluate_run(args.run_dir, eval_dataset, args.n_episodes, seed)
    print(f"Debate pair: {result['eval/mean_reward_raw']:.3f} ± {result['eval/std_reward_raw']:.3f}")


if __name__ == "__main__":
    main()
