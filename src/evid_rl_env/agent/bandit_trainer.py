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


class BanditTrainer:
    def __init__(self, env, policy, config, episodes=50, exp_name="exp", seed=42, use_wandb=False, eval_dataset=None, eval_every=10):
        import random, numpy as np
        random.seed(seed); np.random.seed(seed)
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.eval_every = eval_every
        self.reward_rms = RunningMeanStd()
        self.token_penalty = 0.0001
        self.max_tokens_per_episode = 2000
        self.env = env
        self.policy = policy
        self.config = config
        self.episodes = episodes
        self.seed = seed
        self.tracker = ExperimentTracker(exp_name)

        wandb_config = {
            "algo": "bandit",
            "policy_type": self.policy.__class__.__name__,
            "episodes": self.episodes,
            "seed": seed,
            "alpha": getattr(self.config, "alpha", None),
            "state_dim": getattr(self.policy, "state_dim", None),
        }

        self.tracker.save_config(wandb_config)

        # AUDIT FIX: construct a separate ClaimEnv from eval_dataset rather than
        # reusing the training env; the training env's internal episode state (reset()
        # cursor, reward normalizer exposure) must never bleed into eval measurements
        if eval_dataset is not None:
            from evid_rl_env.environment.environment import ClaimEnv
            eval_env = ClaimEnv(eval_dataset)
            assert eval_env.dataset is not self.env.dataset, (
                "BanditTrainer: eval_env.dataset and env.dataset must be different "
                "objects — passing the same dataset to both leaks eval data into training."
            )
            print("BanditTrainer: eval env isolated from training env")
            self.evaluator = Evaluator(eval_env, policy)
        else:
            self.evaluator = None

        if self.use_wandb:
            wandb.init(project="evid-rl", name=exp_name, config=wandb_config)

        self.rl = LinUCBBandit(
            n_actions=len(policy.actions),
            d=policy.state_dim,
            alpha=config.alpha
        )

    def train(self):
        for ep in range(self.episodes):

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

                # policy used only for payload generation
                _, payload, _ = self.policy.act(state)

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

                    # POLICY INFO
                    "action_probs": self.policy.last_probs.tolist(),
                    "policy_type": self.policy.__class__.__name__,

                    # VALUE FUNCTION (Actor-Critic)
                    "value_estimate": None,

                    # ADVANTAGE SIGNAL
                    "advantage": None,

                    # LLM SCORES
                    "llm_scores": llm_scores,

                    # TOKEN USAGE
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

            # LOGGING
            metrics = {
                "episode": ep,
                "reward": total_reward,
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
                eval_metrics = self.evaluator.evaluate()
                print(f"  Eval | reward {eval_metrics['eval/mean_reward']:.3f} ± {eval_metrics['eval/std_reward']:.3f}")
                if self.use_wandb:
                    wandb.log(eval_metrics, step=ep)

        if self.use_wandb:
            wandb.finish()
