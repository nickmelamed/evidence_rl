"""
evid-gold-eval: Standalone gold-judge evaluation of a trained RL policy.

Re-scores a saved checkpoint's held-out-split trajectories with a held-out
gold judge (never used in training reward — see agent/gold_evaluator.py),
independent of a live training run. Appends its result to the checkpoint's
experiment dir as gold_eval.jsonl, same shape as the rows base_trainer.py's
EvalMixin writes during training, so both sources feed the same dashboard
chart.
"""

from __future__ import annotations

import argparse
import json
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pathlib import Path

from evid_rl_env.agent.config_loader import load_base_config
from evid_rl_env.cli.eval import _deterministic_split, _find_latest_checkpoint
from evid_rl_env.data.dataset import load_dataset
from evid_rl_env.data.evidence_fetcher import use_snapshot


def _append_gold_eval_jsonl(exp_dir: str, summary: dict) -> str:
    path = os.path.join(exp_dir, "gold_eval.jsonl")
    with open(path, "a") as f:
        f.write(json.dumps(summary) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-score a checkpoint's held-out trajectories with a held-out gold judge."
    )
    parser.add_argument("--checkpoint", default=None,
                         help="Path to a .npz checkpoint. Defaults to the most recently modified experiment.")
    parser.add_argument("--eval-data", default=None,
                         help="Path to a JSON eval dataset. Defaults to the held-out 20%% split from seed_claims.json.")
    parser.add_argument("--judge-model", default=None,
                         help="Training-side judge model (produces the 'proxy' scores). Defaults to ClaimEnv's own default.")
    parser.add_argument("--judge-ensemble-models", nargs="+", default=None,
                         help="If the checkpoint was trained with an ensemble/escalation/debate judge "
                              "(see configs/ppo_ensemble_judge.yaml etc.), pass the same "
                              "judge_ensemble_models list here so the 'proxy' side reconstructs the "
                              "real training architecture instead of a plain single judge. There's no "
                              "way to auto-detect this from the checkpoint file itself — it must match "
                              "what actually trained it.")
    parser.add_argument("--judge-escalation", action="store_true",
                         help="Reconstruct the training judge as an EscalatingJudge (requires "
                              "--judge-ensemble-models). Matches the training config's judge_escalation.")
    parser.add_argument("--judge-escalation-target", default="ensemble", choices=["ensemble", "debate"],
                         help="Which architecture judge_escalation escalates to — matches the training "
                              "config's judge_escalation_target.")
    parser.add_argument("--gold-judge-model", default=None,
                         help="Held-out judge model. Defaults to configs/base.yaml's gold_judge_model.")
    parser.add_argument("--n-episodes", type=int, default=None,
                         help="Defaults to configs/base.yaml's gold_eval_n_episodes.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--evidence-snapshot", type=str, default=None,
                         help="Path to a JSON snapshot from evid-snapshot, for reproducible evidence.")
    args = parser.parse_args()

    if args.evidence_snapshot is not None:
        use_snapshot(args.evidence_snapshot)

    base_cfg = load_base_config()
    seed = args.seed if args.seed is not None else base_cfg.get("seed", 42)
    gold_judge_model = args.gold_judge_model or base_cfg.get("gold_judge_model", "mistralai/Mistral-7B-Instruct-v0.2")
    n_episodes = args.n_episodes or base_cfg.get("gold_eval_n_episodes", 20)

    if args.eval_data:
        with open(args.eval_data) as f:
            eval_dataset = json.load(f)
    else:
        _train, eval_dataset = _deterministic_split(load_dataset())

    checkpoint = args.checkpoint or _find_latest_checkpoint()
    if checkpoint is None:
        print("No checkpoint found under artifacts/experiments/. Run evid-train first or pass --checkpoint.")
        raise SystemExit(1)

    from evid_rl_env.agent.bandit import LinUCBBandit
    from evid_rl_env.agent.policy import ActorCriticPolicy, BanditPolicyWrapper
    import numpy as np

    _peek = np.load(checkpoint, allow_pickle=False)
    ckpt_type = str(_peek.get("type", ["actor_critic"])[0])
    if ckpt_type == "bandit":
        bandit, model_name = LinUCBBandit.load(checkpoint)
        inner = ActorCriticPolicy(
            n_actions=int(_peek["n_actions"][0]), state_dim=int(_peek["d"][0]),
            model_name=model_name, seed=seed,
        )
        policy = BanditPolicyWrapper(bandit, inner)
    else:
        policy = ActorCriticPolicy.load(checkpoint, seed=seed)

    if args.judge_escalation and args.judge_ensemble_models:
        judge_desc = f"escalating -> {args.judge_escalation_target} {args.judge_ensemble_models}"
    elif args.judge_ensemble_models:
        judge_desc = f"ensemble {args.judge_ensemble_models}"
    else:
        judge_desc = args.judge_model or "(ClaimEnv default)"

    print(f"Loaded checkpoint: {checkpoint} (type={ckpt_type})")
    print(f"Gold judge: {gold_judge_model} | training judge: {judge_desc}")
    print(f"Eval split: {len(eval_dataset)} samples | gold subsample: {n_episodes}")

    from evid_rl_env.agent.gold_evaluator import GoldEvaluator
    from evid_rl_env.agent.llm_client import JudgeLLMClient
    from evid_rl_env.environment.environment import ClaimEnv
    from evid_rl_env.judge.llm_judge import LLMJudge
    import random as _random

    _rng_state = _random.getstate()
    _random.seed(seed)
    gold_subset = _random.sample(eval_dataset, min(n_episodes, len(eval_dataset)))
    _random.setstate(_rng_state)

    gold_judge = LLMJudge(
        JudgeLLMClient(model_name=gold_judge_model, seed=seed),
        cache_path="artifacts/cache/gold_judge_cache.sqlite3",
    )
    gold_env = ClaimEnv(
        gold_subset, judge_model=args.judge_model, seed=seed,
        judge_ensemble_models=args.judge_ensemble_models,
        judge_escalation=args.judge_escalation,
        judge_escalation_target=args.judge_escalation_target,
    )
    evaluator = GoldEvaluator(gold_env, policy, [gold_judge], n_episodes=len(gold_subset))

    print("\nRunning gold evaluation...")
    summary = evaluator.evaluate()

    print(f"\n[Gold eval] checkpoint: {checkpoint}")
    print(f"  scored {summary['n_scored']}/{summary['n_episodes']} episodes")
    print(f"  proxy reward: {summary['proxy_reward_mean']}")
    print(f"  gold reward:  {summary['gold_reward_mean']}")
    print(f"  correlation:  {summary['proxy_gold_correlation']}")
    print(f"  accuracy:     {summary['outcome_accuracy']} (hard: {summary['outcome_accuracy_hard']})")
    for dim, vals in summary["dimensions"].items():
        print(f"  {dim:<5} proxy={vals['proxy_mean']:.3f} gold={vals['gold_mean']:.3f} "
              f"|disagreement|={vals['mean_abs_disagreement']:.3f}")

    exp_dir = str(Path(checkpoint).parent)
    row = {"episode": None, "checkpoint": checkpoint, **summary}
    out_path = _append_gold_eval_jsonl(exp_dir, row)
    print(f"\nGold eval results appended -> {out_path}")


if __name__ == "__main__":
    main()
