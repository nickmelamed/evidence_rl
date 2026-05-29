import numpy as np
import json

from evid_rl_env.utils.experiment import ExperimentTracker
from evid_rl_env.agent.bandit import LinUCBBandit
from evid_rl_env.agent.policy_gradient import PolicyGradient
from evid_rl_env.agent.ppo import PPO
from evid_rl_env.environment.actions import Actions, ACTIONS
from evid_rl_env.agent.policy import encode_state


class Trainer:
    def __init__(self, env, policy, config, episodes=50, algo="ppo", exp_name="exp", seed=42):
        import random, numpy as np
        random.seed(seed); np.random.seed(seed)
        self.env = env
        self.policy = policy
        self.config = config
        self.episodes = episodes
        self.algo = algo
        self.seed = seed
        self.tracker = ExperimentTracker(exp_name)

        self.tracker.save_config({
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

            # Bandit
            "alpha": getattr(self.config, "alpha", None),

            # Policy
            "state_dim": getattr(self.policy, "state_dim", None),
        })

        # RL selection
        if algo == "ppo":
            assert hasattr(config, "clip"), "PPOConfig required"
            self.rl = PPO(policy, config)

        elif algo == "pg":
            assert hasattr(config, "lr"), "PGConfig required"
            self.rl = PolicyGradient(policy, config)

        elif algo == "bandit":
            self.rl = LinUCBBandit(
                n_actions=len(policy.actions),
                d=policy.state_dim,
                alpha=config.alpha
            )

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

                # action selection 
                if self.algo == "bandit":
                    x = encode_state(state)
                    action_idx = self.rl.select_action(x)
                    action = self.policy.actions[action_idx]

                    # reuse policy for payload
                    _, payload, _ = self.policy.act(state)

                    prob = None
                    value = None

                else:
                    action, payload, action_idx = self.policy.act(state)

                    prob = self.policy.get_probs(state)[action_idx]
                    value = self.policy.get_value(state)

                # env step 
                next_state, reward, done, info = self.env.step(action, payload)

                llm_scores = info.get("llm_scores", {})
                llm_reward = info.get("llm_reward", 0)

                if isinstance(payload, dict):
                    total_tokens += payload.get("tokens", 0)

                # store unified trajectory 
                trajectory.append({
                    "state": state,
                    "action_idx": action_idx,
                    "reward": reward,
                    "prob": prob,
                    "value": value
                })

                # online bandit update 
                if self.algo == "bandit":
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
                "value_estimate": value,

                # ADVANTAGE SIGNAL
                "advantage": reward - value if value is not None else None,

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

            # bandit already updated online

            # LOGGING
            metrics = {
                "episode": ep,
                "reward": total_reward,
                "num_steps": steps,
                "entropy": self.policy.last_entropy,
                "tokens": total_tokens 
            }

            self.tracker.log_episode(metrics)
            self.tracker.save_trajectory(ep, viz)

            print(f"Ep {ep} | Reward {total_reward:.3f}")