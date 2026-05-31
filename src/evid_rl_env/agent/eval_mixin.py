import csv
import os
import random

import numpy as np

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

_BASELINE_TABLE_ORDER = [
    "random", "majority", "greedy_llm",
    "fewshot_k3", "fewshot_k5", "best_of_5", "imitation",
]

_CSV_COLUMNS = [
    "episode", "rl_mean", "rl_std",
    "random_mean", "majority_mean", "greedy_llm_mean",
    "fewshot_k3_mean", "fewshot_k5_mean", "best_of_5_mean", "imitation_mean",
    "lcs", "ess", "grs", "comp", "delta_vs_greedy_llm",
]


class EvalMixin:
    """Shared eval-round logic for Trainer and BanditTrainer.

    Requires the host class to set: self.evaluator, self.baselines,
    self.baseline_n_episodes, self.seed, self.tracker, self.use_wandb,
    self._eval_csv_path.
    """

    def _run_eval_round(self, ep: int) -> None:
        # AUDIT FIX: save and restore global RNG state so baseline evaluations
        # (RandomBaseline, MajorityBaseline, etc.) don't consume training RNG entropy
        # and cause training to produce different results depending on eval frequency.
        # Also seed eval deterministically so eval results are reproducible across runs.
        _rng_state = random.getstate()
        _np_rng_state = np.random.get_state()
        random.seed(self.seed ^ (ep + 1))
        np.random.seed(self.seed ^ (ep + 1))
        try:
            self._run_eval_round_inner(ep)
        finally:
            random.setstate(_rng_state)
            np.random.set_state(_np_rng_state)

    def _run_eval_round_inner(self, ep: int) -> None:
        rl_metrics = self.evaluator.evaluate()
        rl_raw = rl_metrics["eval/mean_reward_raw"]
        rl_raw_std = rl_metrics["eval/std_reward_raw"]
        rl_norm = rl_metrics["eval/mean_reward"]
        rl_norm_std = rl_metrics["eval/std_reward"]

        baseline_results = {}
        for name, bl in self.baselines.items():
            baseline_results[name] = bl.run(self.baseline_n_episodes)

        ref = baseline_results.get("greedy_llm", {}).get("mean_reward", 0.0)

        print(f"[Eval ep {ep + 1}]")
        for name in _BASELINE_TABLE_ORDER:
            if name not in baseline_results:
                continue
            r = baseline_results[name]
            mean, std = r["mean_reward"], r["std_reward"]
            if name == "greedy_llm":
                print(f"  {name:<14} {mean:.2f} ± {std:.2f}  (baseline)")
            else:
                print(f"  {name:<14} {mean:.2f} ± {std:.2f}  Δ {mean - ref:+.2f}")

        rl_delta = rl_raw - ref
        print(
            f"  {'RL (greedy)':<14} {rl_raw:.2f} ± {rl_raw_std:.2f}"
            f"  Δ {rl_delta:+.2f}  ← target"
        )
        print(f"  (RL normalized:  {rl_norm:.3f} ± {rl_norm_std:.3f})")

        self._append_eval_csv(ep, rl_raw, rl_raw_std, baseline_results, ref, rl_metrics)
        self.tracker.log_eval(ep + 1, rl_norm, rl_norm_std)

        if self.use_wandb:
            combined = dict(rl_metrics)
            for name, r in baseline_results.items():
                combined[f"baseline/{name}/mean_reward"] = r["mean_reward"]
            wandb.log(combined, step=ep)

    def _append_eval_csv(
        self,
        ep: int,
        rl_raw: float,
        rl_raw_std: float,
        baseline_results: dict,
        ref_mean: float,
        rl_metrics: dict,
    ) -> None:
        os.makedirs(os.path.dirname(self._eval_csv_path), exist_ok=True)
        write_header = not os.path.exists(self._eval_csv_path)

        def _get(name):
            return baseline_results.get(name, {}).get("mean_reward", "")

        row = {
            "episode":           ep + 1,
            "rl_mean":           rl_raw,
            "rl_std":            rl_raw_std,
            "random_mean":       _get("random"),
            "majority_mean":     _get("majority"),
            "greedy_llm_mean":   _get("greedy_llm"),
            "fewshot_k3_mean":   _get("fewshot_k3"),
            "fewshot_k5_mean":   _get("fewshot_k5"),
            "best_of_5_mean":    _get("best_of_5"),
            "imitation_mean":    _get("imitation"),
            "lcs":               rl_metrics.get("eval/llm_LCS", ""),
            "ess":               rl_metrics.get("eval/llm_ESS", ""),
            "grs":               rl_metrics.get("eval/llm_GRS", ""),
            "comp":              rl_metrics.get("eval/llm_COMP", ""),
            "delta_vs_greedy_llm": rl_raw - ref_mean,
        }

        with open(self._eval_csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
