import os

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from evid_rl_env.agent.bandit import LinUCBBandit
from evid_rl_env.agent.base_trainer import BaseTrainer
from evid_rl_env.agent.policy import encode_state


class BanditTrainer(BaseTrainer):
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
        baseline_n_episodes=3,
        curriculum=None,
    ):
        if eval_dataset is not None:
            assert eval_dataset is not env.dataset, (
                "BanditTrainer: eval_dataset and env.dataset must be different "
                "objects — passing the same dataset to both leaks eval data into training."
            )
            print("BanditTrainer: eval env isolated from training env")

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
                "algo": "bandit",
                "alpha": getattr(config, "alpha", None),
            },
        )

        self.rl = LinUCBBandit(
            n_actions=len(policy.actions),
            d=policy.state_dim,
            alpha=config.alpha
        )

    # Training Loop 

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

                reward, done = self._apply_token_budget(total_tokens, reward, done)

                total_reward_raw += reward
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

            self._update_curriculum(total_reward)

            metrics = self._build_episode_metrics(
                ep, total_reward, total_reward_raw, steps, total_tokens, viz
            )

            if self.use_wandb:
                wandb.log(metrics, step=ep)

            self.tracker.log_episode(metrics)
            self.tracker.save_trajectory(ep, viz)

            print(f"Ep {ep} | Reward {total_reward:.3f}")

            if self.evaluator is not None and (ep + 1) % self.eval_every == 0:
                self._run_eval_round(ep)

        checkpoint_path = os.path.join(self.tracker.base_dir, "policy")
        model_name = getattr(getattr(self.policy, "llm", None), "model_name", self.config.actor_model)
        self.rl.save(checkpoint_path, model_name=model_name)
        print(f"Checkpoint saved: {checkpoint_path}.npz")

        self._finish()
