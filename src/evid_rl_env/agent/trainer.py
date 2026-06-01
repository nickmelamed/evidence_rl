# For bandit training use BanditTrainer in bandit_trainer.py
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
from evid_rl_env.agent.eval_mixin import EvalMixin
from evid_rl_env.agent.evaluator import Evaluator
from evid_rl_env.utils.running_stats import RunningMeanStd
from evid_rl_env.agent.policy_gradient import PolicyGradient
from evid_rl_env.agent.ppo import PPO
from evid_rl_env.environment.actions import ACTIONS
from evid_rl_env.agent.baseline import (
    RandomBaseline,
    MajorityBaseline,
    GreedyLLMBaseline,
    FewShotLLMBaseline,
    BestOfNBaseline,
    ImitationBaseline,
)

_TRAJ_PATH = "data/trajectories.jsonl"
_EVAL_CSV_PATH = "logs/eval_metrics.csv"


class Trainer(EvalMixin):
    def __init__(
        self,
        env,
        policy,
        config,
        episodes=50,
        algo="ppo",
        exp_name="exp",
        seed=42,
        use_wandb=False,
        eval_dataset=None,
        eval_every=25,
        baseline_n_episodes=3,
        curriculum=None,
    ):
        import random, numpy as np
        random.seed(seed); np.random.seed(seed)

        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.eval_every = eval_every
        self.baseline_n_episodes = baseline_n_episodes
        self.reward_rms = RunningMeanStd()
        self.token_penalty = 0.0001  # penalty per token used
        self.max_tokens_per_episode = 2000
        self.env = env
        self.policy = policy
        self.config = config
        self.episodes = episodes
        self.algo = algo
        self.seed = seed
        self.curriculum = curriculum
        # Full training dataset for dynamic curriculum sampling each episode.
        # env.dataset may be replaced per-episode, so we keep a stable reference.
        self._train_dataset = list(env.dataset)
        self.tracker = ExperimentTracker(exp_name)
        self._eval_csv_path = _EVAL_CSV_PATH

        wandb_config = {
            "algo": self.algo,
            "policy_type": self.policy.__class__.__name__,
            "episodes": self.episodes,
            "seed": seed,

            # RL config
            "lr": getattr(self.config, "lr", None),
            "gamma": getattr(self.config, "gamma", None),

            # PPO
            "clip": getattr(self.config, "clip", None),
            "entropy_coef": getattr(self.config, "entropy_coef", None),
            "value_coef": getattr(self.config, "value_coef", None),

            # Policy
            "state_dim": getattr(self.policy, "state_dim", None),
            "eval_every": self.eval_every,
        }

        self.tracker.save_config(wandb_config)

        # RL evaluator — receives the training normalizer for reward scale alignment.
        # The normalizer is read-only inside Evaluator (never updated there).
        self.evaluator = None
        self.baselines = {}
        self.llm_client = getattr(self.policy, "llm", None)

        if eval_dataset is not None:
            from evid_rl_env.environment.environment import ClaimEnv
            eval_env = ClaimEnv(eval_dataset)
            self.evaluator = Evaluator(
                eval_env, policy, reward_normalizer=self.reward_rms
            )

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

        # RL algorithm
        if algo == "ppo":
            assert hasattr(config, "clip"), "PPOConfig required"
            self.rl = PPO(policy, config)

        elif algo == "pg":
            assert hasattr(config, "lr"), "PGConfig required"
            self.rl = PolicyGradient(policy, config)

        else:
            raise ValueError(f"Unknown algo: {algo}")

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

            trajectory = []
            viz = []

            while not done:

                steps += 1

                if steps == 1:
                    if hasattr(self.policy, "reset_episode_cache"):
                        self.policy.reset_episode_cache()

                action, payload, action_idx = self.policy.act(state)

                prob = self.policy.last_probs[action_idx]   # already computed inside act()
                value = self.policy.get_value(state)

                # env step
                next_state, reward, done, info = self.env.step(action, payload)

                llm_scores = info.get("llm_scores", {})
                llm_reward = info.get("llm_reward", 0)

                if isinstance(payload, dict):
                    ep_tokens = payload.get("tokens", 0)
                    total_tokens += ep_tokens
                    reward -= self.token_penalty * ep_tokens

                self.reward_rms.update([reward])
                reward = float(self.reward_rms.normalize(reward))

                # store unified trajectory
                trajectory.append({
                    "state": state,
                    "action_idx": action_idx,
                    "reward": reward,
                    "prob": prob,
                    "value": value
                })

                viz.append({
                    "step": steps,

                    "action": action,
                    "action_idx": action_idx,

                    "reward": reward,
                    "llm_reward": llm_reward,

                    "entropy": self.policy.last_entropy,

                    # POLICY INFO
                    "action_probs": self.policy.last_probs.tolist(),
                    "policy_type": self.policy.__class__.__name__,

                    # VALUE FUNCTION (Actor-Critic)
                    "value_estimate": value,

                    # ADVANTAGE SIGNAL
                    "advantage": reward - value if value is not None else None,

                    # LLM SCORES
                    "llm_scores": llm_scores,

                    # TOKEN USAGE
                    "tokens": payload.get("tokens", 0) if isinstance(payload, dict) else 0,

                    "action_names": ACTIONS,
                    "argument": payload.get("argument", "") if isinstance(payload, dict) else "",
                    "action_payload": payload if isinstance(payload, dict) else {},

                    "selected_ids": [e.id for e in next_state.selected_evidence],
                    "claim": state.claim,
                    "evidence_pool": [
                        {"id": e.id, "text": e.text}
                        for e in next_state.evidence_pool
                    ]
                })

                state = next_state
                total_reward += reward

            # episode-level policy update
            if self.algo == "ppo":
                self.rl.update([
                    (t["state"], t["action_idx"], t["prob"], t["reward"], t["value"])
                    for t in trajectory
                ])

            elif self.algo == "pg":
                self.rl.update([
                    (t["state"], t["action_idx"], t["reward"])
                    for t in trajectory
                ])

            if self.curriculum is not None:
                # sigmoid maps normalized reward → [0, 1] performance signal
                perf = 1.0 / (1.0 + math.exp(max(-50.0, min(50.0, -total_reward / 2.0))))
                self.curriculum.update(perf)

            # LOGGING
            metrics = {
                "episode": ep,
                "reward": total_reward,
                "reward_raw": total_reward,
                "curriculum_level": self.curriculum.level if self.curriculum is not None else None,
                "num_steps": steps,
                "entropy": self.policy.last_entropy,
                "tokens": total_tokens
            }

            if self.algo == "pg":
                metrics["pg_lr"] = self.rl.current_lr

            action_dist = {}
            for t in viz:
                a = t["action"]
                action_dist[a] = action_dist.get(a, 0) + 1

            metrics["action_dist"] = {k: v / max(steps, 1) for k, v in action_dist.items()}
            for action_name, frac in metrics.get("action_dist", {}).items():
                safe_key = "action_dist." + action_name.replace(" ", "_")
                metrics[safe_key] = frac
            metrics.pop("action_dist", None)
            metrics["entropy"] = self.policy.last_entropy

            if self.use_wandb:
                wandb.log(metrics, step=ep)

            self.tracker.log_episode(metrics)
            self.tracker.save_trajectory(ep, viz)

            print(f"Ep {ep} | Reward {total_reward:.3f}")

            if self.evaluator is not None and (ep + 1) % self.eval_every == 0:
                self._run_eval_round(ep)

        # Save final policy checkpoint alongside the experiment artefacts
        checkpoint_path = os.path.join(self.tracker.base_dir, "policy")
        self.policy.save(checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}.npz")

        if self.use_wandb:
            wandb.finish()
