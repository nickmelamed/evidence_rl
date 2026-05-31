import csv
import os
import random
from pathlib import Path

import numpy as np

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from evid_rl_env.utils.experiment import ExperimentTracker
from evid_rl_env.agent.evaluator import Evaluator
from evid_rl_env.utils.running_stats import RunningMeanStd
from evid_rl_env.agent.bandit import LinUCBBandit
from evid_rl_env.agent.policy import encode_state
from evid_rl_env.agent.baseline import (
    RandomBaseline,
    MajorityBaseline,
    GreedyLLMBaseline,
    FewShotLLMBaseline,
    BestOfNBaseline,
    ImitationBaseline,
)

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

_TRAJ_PATH = "data/trajectories.jsonl"
_EVAL_CSV_PATH = "logs/eval_metrics.csv"


class BanditTrainer:
    def __init__(
        self,
        env,
        policy,
        config,
        episodes=50,
        exp_name="exp",
        seed=42,
        use_wandb=False,
        eval_dataset=None,
        eval_every=25,
        baseline_n_episodes=5,
        curriculum=None,
    ):
        random.seed(seed)
        np.random.seed(seed)

        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.eval_every = eval_every
        self.baseline_n_episodes = baseline_n_episodes
        self.reward_rms = RunningMeanStd()
        self.token_penalty = 0.0001
        self.max_tokens_per_episode = 2000
        self.env = env
        self.policy = policy
        self.config = config
        self.episodes = episodes
        self.curriculum = curriculum
        self._train_dataset = list(env.dataset)
        self.seed = seed
        self.tracker = ExperimentTracker(exp_name)
        self._eval_csv_path = _EVAL_CSV_PATH

        wandb_config = {
            "algo": "bandit",
            "policy_type": self.policy.__class__.__name__,
            "episodes": self.episodes,
            "seed": seed,
            "alpha": getattr(self.config, "alpha", None),
            "state_dim": getattr(self.policy, "state_dim", None),
            "eval_every": self.eval_every,
        }

        self.tracker.save_config(wandb_config)

        self.evaluator = None
        self.baselines = {}
        self.llm_client = getattr(self.policy, "llm", None)

        if eval_dataset is not None:
            from evid_rl_env.environment.environment import ClaimEnv
            eval_env = ClaimEnv(eval_dataset)
            assert eval_env.dataset is not self.env.dataset, (
                "BanditTrainer: eval_env.dataset and env.dataset must be different "
                "objects — passing the same dataset to both leaks eval data into training."
            )
            print("BanditTrainer: eval env isolated from training env")
            self.evaluator = Evaluator(eval_env, policy, reward_normalizer=self.reward_rms)

            if self.llm_client is not None:
                train_dataset = env.dataset
                self.baselines = {
                    "random":     RandomBaseline(eval_dataset),
                    "majority":   MajorityBaseline(eval_dataset),
                    "greedy_llm": GreedyLLMBaseline(eval_dataset, self.llm_client),
                    "fewshot_k3": FewShotLLMBaseline(eval_dataset, train_dataset, self.llm_client, k=3),
                    "fewshot_k5": FewShotLLMBaseline(eval_dataset, train_dataset, self.llm_client, k=5),
                    "best_of_5":  BestOfNBaseline(eval_dataset, self.llm_client, n=5),
                }
                if Path(_TRAJ_PATH).exists():
                    self.baselines["imitation"] = ImitationBaseline(eval_dataset, _TRAJ_PATH)

        print(
            f"Train: {len(env.dataset)} samples | "
            f"Eval: {len(eval_dataset) if eval_dataset else 0} samples | "
            f"Baselines: {list(self.baselines)}"
        )

        if self.use_wandb:
            wandb.init(project="evid-rl", name=exp_name, config=wandb_config)

        self.rl = LinUCBBandit(
            n_actions=len(policy.actions),
            d=policy.state_dim,
            alpha=config.alpha
        )

    # ------------------------------------------------------------------
    # Eval round — identical structure to Trainer._run_eval_round
    # ------------------------------------------------------------------

    def _run_eval_round(self, ep: int) -> None:
        _rng_state = random.getstate()
        _np_rng_state = np.random.get_state()
        random.seed(self.seed ^ (ep + 1))
        np.random.seed(self.seed ^ (ep + 1))
        try:
            self.__run_eval_round_inner(ep)
        finally:
            random.setstate(_rng_state)
            np.random.set_state(_np_rng_state)

    def __run_eval_round_inner(self, ep: int) -> None:
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
            "episode":          ep + 1,
            "rl_mean":          rl_raw,
            "rl_std":           rl_raw_std,
            "random_mean":      _get("random"),
            "majority_mean":    _get("majority"),
            "greedy_llm_mean":  _get("greedy_llm"),
            "fewshot_k3_mean":  _get("fewshot_k3"),
            "fewshot_k5_mean":  _get("fewshot_k5"),
            "best_of_5_mean":   _get("best_of_5"),
            "imitation_mean":   _get("imitation"),
            "lcs":              rl_metrics.get("eval/llm_LCS", ""),
            "ess":              rl_metrics.get("eval/llm_ESS", ""),
            "grs":              rl_metrics.get("eval/llm_GRS", ""),
            "comp":             rl_metrics.get("eval/llm_COMP", ""),
            "delta_vs_greedy_llm": rl_raw - ref_mean,
        }

        with open(self._eval_csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self):
        import math
        for ep in range(self.episodes):

            if self.curriculum is not None:
                self.env.dataset = [self.curriculum.sample(self._train_dataset)]

            state = self.env.reset()
            done = False

            total_reward = 0
            total_tokens = 0
            steps = 0

            viz = []

            while not done:

                steps += 1

                x = encode_state(state)
                action_idx = self.rl.select_action(x)
                action = self.policy.actions[action_idx]

                _, payload, _ = self.policy.act(state, force_action_idx=action_idx)

                next_state, reward, done, info = self.env.step(action, payload)

                llm_scores = info.get("llm_scores", {})
                llm_reward = info.get("llm_reward", 0)

                if isinstance(payload, dict):
                    ep_tokens = payload.get("tokens", 0)
                    total_tokens += ep_tokens
                    reward -= self.token_penalty * ep_tokens

                self.reward_rms.update([reward])
                reward = float(self.reward_rms.normalize(reward))

                self.rl.update(action_idx, x, reward)

                viz.append({
                    "step": steps,

                    "action": action,
                    "action_idx": action_idx,

                    "reward": reward,
                    "llm_reward": llm_reward,

                    "entropy": self.policy.last_entropy,

                    "action_probs": self.policy.last_probs.tolist(),
                    "policy_type": self.policy.__class__.__name__,

                    "value_estimate": None,
                    "advantage": None,

                    "llm_scores": llm_scores,

                    "tokens": payload.get("tokens", 0) if isinstance(payload, dict) else 0,

                    "selected_ids": [e.id for e in next_state.selected_evidence],
                    "claim": state.claim,
                    "evidence_pool": [
                        {"id": e.id, "text": e.text}
                        for e in next_state.evidence_pool
                    ]
                })

                state = next_state
                total_reward += reward

            if self.curriculum is not None:
                perf = 1.0 / (1.0 + math.exp(max(-50.0, min(50.0, -total_reward / 2.0))))
                self.curriculum.update(perf)

            metrics = {
                "episode": ep,
                "reward": total_reward,
                "curriculum_level": self.curriculum.level if self.curriculum is not None else None,
                "num_steps": steps,
                "entropy": self.policy.last_entropy,
                "tokens": total_tokens
            }

            action_dist = {}
            for t in viz:
                a = t["action"]
                action_dist[a] = action_dist.get(a, 0) + 1

            metrics["action_dist"] = {k: v / max(steps, 1) for k, v in action_dist.items()}
            metrics["entropy"] = self.policy.last_entropy

            if self.use_wandb:
                wandb.log(metrics, step=ep)

            self.tracker.log_episode(metrics)
            self.tracker.save_trajectory(ep, viz)

            print(f"Ep {ep} | Reward {total_reward:.3f}")

            if self.evaluator is not None and (ep + 1) % self.eval_every == 0:
                self._run_eval_round(ep)

        checkpoint_path = os.path.join(self.tracker.base_dir, "policy")
        model_name = getattr(getattr(self.policy, "llm", None), "model_name", "google/gemma-2-2b-it")
        self.rl.save(checkpoint_path, model_name=model_name)
        print(f"Checkpoint saved: {checkpoint_path}.npz")

        if self.use_wandb:
            wandb.finish()
