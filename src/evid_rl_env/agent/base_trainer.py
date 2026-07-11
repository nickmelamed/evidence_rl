import math
import random
from pathlib import Path

import numpy as np

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from evid_rl_env.agent.baseline import (
    BestOfNBaseline,
    FewShotLLMBaseline,
    GreedyLLMBaseline,
    ImitationBaseline,
    MajorityBaseline,
    RandomBaseline,
)
from evid_rl_env.agent.eval_mixin import EvalMixin
from evid_rl_env.agent.evaluator import Evaluator
from evid_rl_env.utils.experiment import ExperimentTracker
from evid_rl_env.utils.running_stats import RunningMeanStd

_TRAJ_PATH = "data/trajectories.jsonl"
_EVAL_CSV_PATH = "logs/eval_metrics.csv"


def build_standard_baselines(eval_dataset, train_dataset, llm_client, fewshot_selection_mode="random"):
    """The fixed set of baselines every trainer evaluates against during training."""
    baselines = {
        "random":     RandomBaseline(eval_dataset),
        "majority":   MajorityBaseline(eval_dataset),
        "greedy_llm": GreedyLLMBaseline(eval_dataset, llm_client),
        "fewshot_k3": FewShotLLMBaseline(
            eval_dataset, train_dataset, llm_client, k=3, selection_mode=fewshot_selection_mode
        ),
        "fewshot_k5": FewShotLLMBaseline(
            eval_dataset, train_dataset, llm_client, k=5, selection_mode=fewshot_selection_mode
        ),
        "best_of_5":  BestOfNBaseline(eval_dataset, llm_client, n=5),
    }
    if Path(_TRAJ_PATH).exists():
        baselines["imitation"] = ImitationBaseline(eval_dataset, _TRAJ_PATH)
    return baselines


class BaseTrainer(EvalMixin):
    """Shared setup and per-episode helpers for Trainer (PPO/PG) and BanditTrainer.

    The two subclasses genuinely differ in action-selection and RL-update
    mechanics, so `train()` stays subclass-specific — this holds only the
    byte-identical (or near-identical) scaffolding around it.
    """

    def _init_common(
        self,
        env,
        policy,
        config,
        episodes,
        exp_name,
        seed,
        use_wandb,
        eval_dataset,
        eval_every,
        baseline_n_episodes,
        curriculum,
        extra_wandb_config=None,
    ):
        random.seed(seed)
        np.random.seed(seed)

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
        self.seed = seed
        self.curriculum = curriculum
        # Full training dataset for dynamic curriculum sampling each episode.
        # env.dataset may be replaced per-episode, so we keep a stable reference.
        self._train_dataset = list(env.dataset)
        self.tracker = ExperimentTracker(exp_name)
        self._eval_csv_path = _EVAL_CSV_PATH

        wandb_config = {
            "policy_type": self.policy.__class__.__name__,
            "episodes": self.episodes,
            "seed": seed,
            "state_dim": getattr(self.policy, "state_dim", None),
            "eval_every": self.eval_every,
        }
        if extra_wandb_config:
            wandb_config.update(extra_wandb_config)

        self.tracker.save_config(wandb_config)

        # RL evaluator — receives the training normalizer for reward scale alignment.
        # The normalizer is read-only inside Evaluator (never updated there).
        self.evaluator = None
        self.baselines = {}
        self.llm_client = getattr(self.policy, "llm", None)

        if eval_dataset is not None:
            from evid_rl_env.environment.environment import ClaimEnv
            eval_env = ClaimEnv(eval_dataset, judge_model=getattr(config, "judge_model", None))
            self.evaluator = Evaluator(
                eval_env, policy, reward_normalizer=self.reward_rms
            )

            if self.llm_client is not None:
                self.baselines = build_standard_baselines(
                    eval_dataset, env.dataset, self.llm_client,
                    fewshot_selection_mode=getattr(config, "fewshot_selection_mode", "random"),
                )

        print(
            f"Train: {len(env.dataset)} samples | "
            f"Eval: {len(eval_dataset) if eval_dataset else 0} samples | "
            f"Baselines: {list(self.baselines)}"
        )

        if self.use_wandb:
            wandb.init(project="evid-rl", name=exp_name, config=wandb_config)

    def _update_curriculum(self, total_reward: float) -> None:
        """Map normalized episode reward to a [0,1] performance signal via a
        sigmoid and feed it to the curriculum, if one is configured."""
        if self.curriculum is None:
            return
        perf = 1.0 / (1.0 + math.exp(max(-50.0, min(50.0, -total_reward / 2.0))))
        self.curriculum.update(perf)

    def _apply_token_budget(self, total_tokens: float, reward: float, done: bool) -> tuple:
        """Force-end the episode, like the environment's own step-limit does,
        once the per-episode token budget is exceeded — previously this field
        was set but never enforced."""
        if not done and total_tokens >= self.max_tokens_per_episode:
            return -0.2, True
        return reward, done

    def _build_episode_metrics(
        self,
        ep: int,
        total_reward: float,
        total_reward_raw: float,
        steps: int,
        total_tokens: float,
        viz: list,
        extra: dict = None,
    ) -> dict:
        """Per-episode metrics dict, including the dotted action_dist.<action>
        keys ExperimentTracker.FIXED_FIELDS expects for the CSV."""
        metrics = {
            "episode": ep,
            "reward": total_reward,
            "reward_raw": total_reward_raw,
            "curriculum_level": self.curriculum.level if self.curriculum is not None else None,
            "num_steps": steps,
            "entropy": self.policy.last_entropy,
            "tokens": total_tokens,
        }
        if extra:
            metrics.update(extra)

        action_dist = {}
        for t in viz:
            a = t["action"]
            action_dist[a] = action_dist.get(a, 0) + 1
        action_dist = {k: v / max(steps, 1) for k, v in action_dist.items()}
        for action_name, frac in action_dist.items():
            metrics["action_dist." + action_name.replace(" ", "_")] = frac

        return metrics

    def _finish(self) -> None:
        if self.use_wandb:
            wandb.finish()
