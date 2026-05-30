"""
evid-eval: Standalone evaluation of a trained RL policy against baselines.

Loads a saved checkpoint, runs the RL policy and the requested baselines on
a held-out eval split, and prints the same formatted comparison table as the
training loop.

Exit codes:
  0 — RL policy beats greedy_llm (or greedy_llm was not requested)
  1 — RL policy does not beat greedy_llm  (usable in CI)
"""

from __future__ import annotations

import argparse
import json
import random
import sys

from evid_rl_env.data.dataset import load_dataset
from evid_rl_env.environment.environment import ClaimEnv
from evid_rl_env.agent.policy import ActorCriticPolicy
from evid_rl_env.agent.evaluator import Evaluator
from evid_rl_env.agent.baseline import (
    RandomBaseline,
    MajorityBaseline,
    GreedyLLMBaseline,
    FewShotLLMBaseline,
    BestOfNBaseline,
    ImitationBaseline,
)

_TABLE_ORDER = [
    "random", "majority", "greedy_llm",
    "fewshot_k3", "fewshot_k5", "best_of_5", "imitation",
]

_AVAILABLE = frozenset(_TABLE_ORDER)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _deterministic_split(dataset: list) -> tuple[list, list]:
    """Apply the same seed-42 80/20 split used during training."""
    # AUDIT FIX: save/restore global random state so seeding to 42 for the split
    # does not permanently alter the RNG sequence seen by downstream baseline runs
    _saved = random.getstate()
    random.seed(42)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    cut = int(0.8 * len(dataset))
    train = [dataset[i] for i in indices[:cut]]
    eval_ = [dataset[i] for i in indices[cut:]]
    random.setstate(_saved)
    return train, eval_


def _load_datasets(eval_data_path: str | None) -> tuple[list, list | None]:
    """Return (eval_dataset, train_dataset).  train_dataset is None when
    eval_data_path is provided explicitly (train split then unavailable)."""
    if eval_data_path:
        with open(eval_data_path) as f:
            return json.load(f), None
    train, eval_ = _deterministic_split(load_dataset())
    return eval_, train


# ---------------------------------------------------------------------------
# Baseline instantiation
# ---------------------------------------------------------------------------

def _build_baselines(
    requested: list[str],
    eval_dataset: list,
    train_dataset: list | None,
    llm_client,
    trajectories_path: str | None,
) -> dict:
    baselines = {}
    llm_baselines = {"greedy_llm", "fewshot_k3", "fewshot_k5", "best_of_5"}
    fewshot_baselines = {"fewshot_k3", "fewshot_k5"}

    for name in requested:
        if name not in _AVAILABLE:
            print(f"  [warn] unknown baseline '{name}' — skipped")
            continue

        if name in llm_baselines and llm_client is None:
            print(f"  [warn] '{name}' requires an LLM client (policy has no .llm) — skipped")
            continue

        if name in fewshot_baselines and train_dataset is None:
            print(f"  [warn] '{name}' requires a train split (use --eval-data with the full split) — skipped")
            continue

        if name == "imitation" and trajectories_path is None:
            print(f"  [warn] 'imitation' requires --trajectories — skipped")
            continue

        if name == "random":
            baselines[name] = RandomBaseline(eval_dataset)
        elif name == "majority":
            baselines[name] = MajorityBaseline(eval_dataset)
        elif name == "greedy_llm":
            baselines[name] = GreedyLLMBaseline(eval_dataset, llm_client)
        elif name == "fewshot_k3":
            baselines[name] = FewShotLLMBaseline(eval_dataset, train_dataset, llm_client, k=3)
        elif name == "fewshot_k5":
            baselines[name] = FewShotLLMBaseline(eval_dataset, train_dataset, llm_client, k=5)
        elif name == "best_of_5":
            baselines[name] = BestOfNBaseline(eval_dataset, llm_client, n=5)
        elif name == "imitation":
            baselines[name] = ImitationBaseline(eval_dataset, trajectories_path)

    return baselines


# ---------------------------------------------------------------------------
# Table printing (mirrors trainer._run_eval_round)
# ---------------------------------------------------------------------------

def _print_table(label: str, baseline_results: dict, rl_raw: float, rl_std: float) -> float | None:
    """Print the comparison table and return the greedy_llm mean (or None)."""
    ref = baseline_results.get("greedy_llm", {}).get("mean_reward")

    print(f"\n[{label}]")
    for name in _TABLE_ORDER:
        if name not in baseline_results:
            continue
        r = baseline_results[name]
        mean, std = r["mean_reward"], r["std_reward"]
        if name == "greedy_llm":
            print(f"  {name:<14} {mean:.2f} ± {std:.2f}  (baseline)")
        elif ref is not None:
            print(f"  {name:<14} {mean:.2f} ± {std:.2f}  Δ {mean - ref:+.2f}")
        else:
            print(f"  {name:<14} {mean:.2f} ± {std:.2f}")

    delta_str = f"  Δ {rl_raw - ref:+.2f}  ← target" if ref is not None else "  ← target"
    print(f"  {'RL (greedy)':<14} {rl_raw:.2f} ± {rl_std:.2f}{delta_str}")

    return ref


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained RL policy against baselines. "
            "Exits with code 1 if RL does not beat greedy_llm."
        )
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a .npz checkpoint saved by the trainer (e.g. artifacts/experiments/ppo_run_*/policy.npz).",
    )
    parser.add_argument(
        "--eval-data",
        default=None,
        help="Path to a JSON eval dataset. Defaults to the held-out 20%% split from seed_claims.json.",
    )
    parser.add_argument(
        "--trajectories",
        default=None,
        help="Path to trajectories.jsonl for the imitation baseline.",
    )
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument(
        "--baselines",
        default="random,greedy_llm,fewshot_k3,best_of_5",
        help="Comma-separated list of baselines to run.",
    )
    parser.add_argument(
        "--greedy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use greedy (argmax) action selection for the RL policy (default: True).",
    )
    # AUDIT FIX: expose --seed so baseline eval results are reproducible; does not
    # affect the train/eval split (always seed-42 to match train.py)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for baseline evaluation runs (default: 0). "
             "The train/eval split always uses seed 42 regardless of this value.",
    )
    # AUDIT FIX (issue 3): surface --annotator-model so evid-eval's interface is
    # consistent with evid-collect; argument is accepted but not wired to any logic yet —
    # it is reserved for future post-hoc re-annotation workflows
    parser.add_argument(
        "--annotator-model",
        default="claude-opus-4-5",
        help=(
            "Anthropic model string for post-hoc re-annotation workflows (default: claude-opus-4-5). "
            "Currently only relevant when 'imitation' is included in --baselines and "
            "re-annotation is being performed. Does not affect the loaded checkpoint or "
            "any other baseline."
        ),
    )
    args = parser.parse_args()

    requested = [b.strip() for b in args.baselines.split(",") if b.strip()]
    unknown = set(requested) - _AVAILABLE
    if unknown:
        print(f"Unknown baselines: {unknown}. Choose from: {sorted(_AVAILABLE)}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    eval_dataset, train_dataset = _load_datasets(args.eval_data)
    print(
        f"Eval: {len(eval_dataset)} samples"
        + (f" | Train: {len(train_dataset)}" if train_dataset else "")
    )

    # ------------------------------------------------------------------
    # Load policy
    # ------------------------------------------------------------------
    policy = ActorCriticPolicy.load(args.checkpoint)
    # AUDIT FIX: set the LLM seed on the loaded policy so transformers.set_seed is
    # called with the --seed value before every generation, making eval reproducible
    if hasattr(policy, "llm"):
        policy.llm.seed = args.seed
    print(
        f"Loaded checkpoint: {args.checkpoint} "
        f"(state_dim={policy.state_dim}, n_actions={policy.n_actions}, "
        f"greedy={args.greedy})"
    )
    # AUDIT FIX: log LLM seed here so eval invocations are auditable from terminal output
    print(f"LLM seed: {args.seed}")
    llm_client = getattr(policy, "llm", None)

    # ------------------------------------------------------------------
    # RL policy evaluation — no reward normalizer (raw rewards for fair
    # comparison with baselines which also report raw)
    # ------------------------------------------------------------------
    print("\nRunning RL policy evaluation...")
    # AUDIT FIX: pass seed so ClaimEnv threads it to JudgeLLMClient for reproducible
    # judge outputs during eval
    eval_env = ClaimEnv(eval_dataset, seed=args.seed)

    if args.greedy:
        # Wrap act to force greedy selection without modifying the policy object
        original_act = policy.act
        policy.act = lambda state, **kw: original_act(state, greedy=True)

    evaluator = Evaluator(eval_env, policy, n_eval_episodes=args.n_episodes, reward_normalizer=None)
    rl_metrics = evaluator.evaluate()

    if args.greedy:
        policy.act = original_act  # restore

    rl_raw = rl_metrics["eval/mean_reward_raw"]
    rl_std = rl_metrics["eval/std_reward_raw"]

    # AUDIT FIX (issue 3): log the annotator-model note whenever imitation is requested
    # so the interface intent is visible in logs even before re-annotation logic is wired in
    if "imitation" in requested:
        print(
            f"[info] --annotator-model='{args.annotator_model}' is set. "
            "Note: annotator model arg is only used for post-hoc re-annotation workflows "
            "— it does not affect the loaded checkpoint or other baselines."
        )

    # ------------------------------------------------------------------
    # Baselines
    # ------------------------------------------------------------------
    print("\nInstantiating and running baselines...")
    baselines = _build_baselines(requested, eval_dataset, train_dataset, llm_client, args.trajectories)

    # AUDIT FIX: seed global RNG before running baselines so results are reproducible
    # across repeated evid-eval invocations with the same --seed value
    random.seed(args.seed)
    import numpy as np
    np.random.seed(args.seed)

    baseline_results = {}
    for name, bl in baselines.items():
        print(f"  Running {name}...")
        baseline_results[name] = bl.run(args.n_episodes)

    # ------------------------------------------------------------------
    # Table + CI gate
    # ------------------------------------------------------------------
    ref = _print_table(
        f"Eval — {args.n_episodes} episodes | checkpoint: {args.checkpoint}",
        baseline_results,
        rl_raw,
        rl_std,
    )

    print()
    if ref is None:
        print("greedy_llm not in requested baselines — CI gate skipped.")
        sys.exit(0)

    if rl_raw > ref:
        print(f"PASS: RL ({rl_raw:.3f}) > greedy_llm ({ref:.3f})")
        sys.exit(0)
    else:
        print(f"FAIL: RL ({rl_raw:.3f}) ≤ greedy_llm ({ref:.3f})")
        sys.exit(1)


if __name__ == "__main__":
    main()
