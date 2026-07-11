# For bandit training use BanditTrainer in bandit_trainer.py
import os

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from evid_rl_env.agent.base_trainer import BaseTrainer
from evid_rl_env.agent.policy_gradient import PolicyGradient
from evid_rl_env.agent.ppo import PPO
from evid_rl_env.environment.actions import ACTIONS


class Trainer(BaseTrainer):
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
        self.algo = algo
        self._init_common(
            env=env,
            policy=policy,
            config=config,
            episodes=episodes,
            exp_name=exp_name,
            seed=seed,
            use_wandb=use_wandb,
            eval_dataset=eval_dataset,
            eval_every=eval_every,
            baseline_n_episodes=baseline_n_episodes,
            curriculum=curriculum,
            extra_wandb_config={
                "algo": self.algo,
                "lr": getattr(config, "lr", None),
                "gamma": getattr(config, "gamma", None),
                "clip": getattr(config, "clip", None),
                "entropy_coef": getattr(config, "entropy_coef", None),
                "value_coef": getattr(config, "value_coef", None),
            },
        )

        # RL algorithm
        if algo == "ppo":
            assert hasattr(config, "clip"), "PPOConfig required"
            self.rl = PPO(policy, config)

        elif algo == "pg":
            assert hasattr(config, "lr"), "PGConfig required"
            self.rl = PolicyGradient(policy, config)

        else:
            raise ValueError(f"Unknown algo: {algo}")

    # Training loop 
    
    def train(self):
        for ep in range(self.episodes):

            if self.curriculum is not None:
                self.env.dataset = [self.curriculum.sample(self._train_dataset)]

            state = self.env.reset()
            done = False

            total_reward = 0.0
            total_reward_raw = 0.0
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

                reward, done = self._apply_token_budget(total_tokens, reward, done)

                total_reward_raw += reward
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

            self._update_curriculum(total_reward)

            extra = {"pg_lr": self.rl.current_lr} if self.algo == "pg" else {}
            metrics = self._build_episode_metrics(
                ep, total_reward, total_reward_raw, steps, total_tokens, viz, extra=extra
            )

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

        self._finish()
