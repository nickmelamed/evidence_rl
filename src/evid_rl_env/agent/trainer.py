# For bandit training use BanditTrainer in bandit_trainer.py
import numpy as np

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from evid_rl_env.utils.experiment import ExperimentTracker
from evid_rl_env.agent.evaluator import Evaluator
from evid_rl_env.utils.running_stats import RunningMeanStd
from evid_rl_env.agent.policy_gradient import PolicyGradient
from evid_rl_env.agent.ppo import PPO
from evid_rl_env.environment.actions import ACTIONS


class Trainer:
    def __init__(self, env, policy, config, episodes=50, algo="ppo", exp_name="exp", seed=42, use_wandb=False, eval_dataset=None, eval_every=10):
        import random, numpy as np
        random.seed(seed); np.random.seed(seed)
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.eval_every = eval_every
        self.reward_rms = RunningMeanStd()
        self.token_penalty = 0.0001  # penalty per token used
        self.max_tokens_per_episode = 2000
        self.env = env
        self.policy = policy
        self.config = config
        self.episodes = episodes
        self.algo = algo
        self.seed = seed
        self.tracker = ExperimentTracker(exp_name)

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
        }

        self.tracker.save_config(wandb_config)
        self.evaluator = Evaluator(env, policy) if eval_dataset is not None else None

        if self.use_wandb:
            wandb.init(project="evid-rl", name=exp_name, config=wandb_config)

        # RL selection
        if algo == "ppo":
            assert hasattr(config, "clip"), "PPOConfig required"
            self.rl = PPO(policy, config)

        elif algo == "pg":
            assert hasattr(config, "lr"), "PGConfig required"
            self.rl = PolicyGradient(policy, config)

        else:
            raise ValueError(f"Unknown algo: {algo}")

    def train(self):
        for ep in range(self.episodes):

            state = self.env.reset()
            done = False

            total_reward = 0
            total_tokens = 0
            steps = 0

            trajectory = []
            viz = []

            while not done:

                steps += 1

                action, payload, action_idx = self.policy.act(state)

                prob = self.policy.get_probs(state)[action_idx]
                value = self.policy.get_value(state)

                # env step
                next_state, reward, done, info = self.env.step(action, payload)

                llm_scores = info.get("llm_scores", {})
                llm_reward = info.get("llm_reward", 0)

                if isinstance(payload, dict):
                    ep_tokens = payload.get("tokens", 0)
                    total_tokens += ep_tokens
                    token_fraction = min(1.0, total_tokens / self.max_tokens_per_episode)
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
                    (
                        t["state"],
                        t["action_idx"],
                        t["prob"],
                        t["reward"],
                        t["value"]
                    )
                    for t in trajectory
                ])

            elif self.algo == "pg":
                self.rl.update([
                    (
                        t["state"],
                        t["action_idx"],
                        t["reward"]
                    )
                    for t in trajectory
                ])

            # LOGGING
            metrics = {
                "episode": ep,
                "reward": total_reward,
                "reward_raw": total_reward,
                "curriculum_level": getattr(self, "curriculum_level", None),
                "num_steps": steps,
                "entropy": self.policy.last_entropy,
                "tokens": total_tokens
            }

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
                eval_metrics = self.evaluator.evaluate()
                print(f"  Eval | reward {eval_metrics['eval/mean_reward']:.3f} ± {eval_metrics['eval/std_reward']:.3f}")
                if self.use_wandb:
                    wandb.log(eval_metrics, step=ep)

        if self.use_wandb:
            wandb.finish()
