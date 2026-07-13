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
    _SharedFewShotExamples,
)
from evid_rl_env.agent.eval_mixin import EvalMixin
from evid_rl_env.agent.evaluator import Evaluator
from evid_rl_env.utils.experiment import ExperimentTracker
from evid_rl_env.utils.running_stats import RunningMeanStd

_TRAJ_PATH = "data/trajectories.jsonl"
_EVAL_CSV_PATH = "logs/eval_metrics.csv"


def build_standard_baselines(eval_dataset, train_dataset, llm_client, fewshot_selection_mode="random"):
    """The fixed set of baselines every trainer evaluates against during training."""
    # Shared (and still fully lazy) example bank: fewshot_k3/fewshot_k5 only
    fewshot_bank = _SharedFewShotExamples(train_dataset)
    baselines = {
        "random":     RandomBaseline(eval_dataset),
        "majority":   MajorityBaseline(eval_dataset),
        "greedy_llm": GreedyLLMBaseline(eval_dataset, llm_client),
        "fewshot_k3": FewShotLLMBaseline(
            eval_dataset, train_dataset, llm_client, k=3, selection_mode=fewshot_selection_mode,
            example_bank=fewshot_bank,
        ),
        "fewshot_k5": FewShotLLMBaseline(
            eval_dataset, train_dataset, llm_client, k=5, selection_mode=fewshot_selection_mode,
            example_bank=fewshot_bank,
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
            # judge architecture — recorded so a run's config.json is
            # self-describing when comparing single/ensemble/escalation
            "judge_model": getattr(config, "judge_model", None),
            "judge_ensemble_models": getattr(config, "judge_ensemble_models", None),
            "judge_escalation": getattr(config, "judge_escalation", False),
            "judge_escalation_target": getattr(config, "judge_escalation_target", "ensemble"),
            "gold_judge_model": getattr(config, "gold_judge_model", None),
        }
        if extra_wandb_config:
            wandb_config.update(extra_wandb_config)

        self.tracker.save_config(wandb_config)

        # RL evaluator — receives the training normalizer for reward scale alignment.
        # The normalizer is read-only inside Evaluator (never updated there).
        self.evaluator = None
        self.baselines = {}
        self.llm_client = getattr(self.policy, "llm", None)
        self.gold_evaluator = None
        self.gold_eval_every = getattr(config, "gold_eval_every", 5)
        self.expensive_baseline_every = getattr(config, "expensive_baseline_every", 5)
        self._eval_round_count = 0

        if eval_dataset is not None:
            from evid_rl_env.environment.environment import ClaimEnv
            eval_env = ClaimEnv(
                eval_dataset, judge_model=getattr(config, "judge_model", None), seed=seed,
                judge_ensemble_models=getattr(config, "judge_ensemble_models", None),
                judge_escalation=getattr(config, "judge_escalation", False),
                judge_escalation_target=getattr(config, "judge_escalation_target", "ensemble"),
            )
            self.evaluator = Evaluator(
                eval_env, policy, reward_normalizer=self.reward_rms
            )

            if self.llm_client is not None:
                self.baselines = build_standard_baselines(
                    eval_dataset, env.dataset, self.llm_client,
                    fewshot_selection_mode=getattr(config, "fewshot_selection_mode", "random"),
                )

            gold_judge_model = getattr(config, "gold_judge_model", None)
            if gold_judge_model:
                self.gold_evaluator = self._build_gold_evaluator(
                    eval_dataset, gold_judge_model,
                    getattr(config, "judge_model", None),
                    getattr(config, "gold_eval_n_episodes", 20),
                    seed,
                )

        print(
            f"Train: {len(env.dataset)} samples | "
            f"Eval: {len(eval_dataset) if eval_dataset else 0} samples | "
            f"Baselines: {list(self.baselines)}"
        )

        if self.use_wandb:
            wandb.init(project="evid-rl", name=exp_name, config=wandb_config)

    def _build_gold_evaluator(self, eval_dataset, gold_judge_model, judge_model, n_episodes, seed):
        """Held-out judge (different model family, never used in training
        reward) plus a fixed-size seeded subsample of eval_dataset, wrapped
        in a GoldEvaluator — see agent/gold_evaluator.py. Kept a small
        subsample regardless of eval_dataset size since gold_judge_model is
        typically 5-10x larger than judge_model and this runs on its own,
        coarser cadence (self.gold_eval_every)."""
        from evid_rl_env.agent.gold_evaluator import GoldEvaluator
        from evid_rl_env.agent.llm_client import JudgeLLMClient
        from evid_rl_env.environment.environment import ClaimEnv
        from evid_rl_env.judge.llm_judge import LLMJudge

        gold_llm_client = JudgeLLMClient(model_name=gold_judge_model, seed=seed)
        gold_judge = LLMJudge(
            gold_llm_client, cache_path="artifacts/cache/gold_judge_cache.sqlite3",
        )

        _rng_state = random.getstate()
        random.seed(seed)
        gold_subset = random.sample(eval_dataset, min(n_episodes, len(eval_dataset)))
        random.setstate(_rng_state)

        gold_env = ClaimEnv(gold_subset, judge_model=judge_model, seed=seed)
        return GoldEvaluator(gold_env, self.policy, [gold_judge], n_episodes=len(gold_subset))

    def _update_curriculum(self, claim_id, task_success) -> None:
        """Feed the episode's task-success signal to the curriculum, keyed by
        the specific claim just attempted (per-claim Prioritized Level
        Replay — see curriculum.py). task_success is the bounded [0,1]
        calibration score from the FINALIZE step's info dict (1 - |confidence
        - true_score|); episodes that never reach FINALIZE (step-limit cutoff,
        token budget) count as 0 — the agent didn't complete the task."""
        if self.curriculum is None:
            return
        perf = task_success if task_success is not None else 0.0
        self.curriculum.update(claim_id, perf)

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
            "curriculum_mean_score": self.curriculum.mean_score if self.curriculum is not None else None,
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
