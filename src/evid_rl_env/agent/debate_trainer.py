import os

import numpy as np

from evid_rl_env.agent.base_trainer import BaseTrainer
from evid_rl_env.agent.ppo import PPO
from evid_rl_env.environment.actions import ACTIONS
from evid_rl_env.environment.debate_env import ROLES, DebateEnv


class DebateTrainer(BaseTrainer):
    """Train a support policy and a contradict policy against a shared arbiter.

    The roles take turns in one ``DebateEnv`` episode. Each role keeps its own
    trajectory and its own PPO update, built from its own turns only. The
    arbiter's final reward goes to both roles, added to each role's last step.

    The per-episode token budget is not enforced, because ending a debate early
    would skip the arbiter's score. The per-token penalty still applies.

    There is no evaluation round here yet. The existing evaluators play one
    policy, so the pair is scored by a separate debate evaluator.
    """

    def __init__(
        self,
        env,
        support_policy,
        contradict_policy,
        config,
        episodes=50,
        exp_name="debate",
        seed=42,
        use_wandb=False,
        curriculum=None,
        turns_per_side=5,
        arbiter_model=None,
    ):
        self.policies = {"support": support_policy, "contradict": contradict_policy}
        self._init_common(
            env=env,
            policy=support_policy,
            config=config,
            episodes=episodes,
            exp_name=exp_name,
            seed=seed,
            use_wandb=use_wandb,
            eval_dataset=None,
            eval_every=0,
            baseline_n_episodes=0,
            curriculum=curriculum,
            extra_wandb_config={
                "algo": "ppo_debate",
                "roles": list(ROLES),
                "turns_per_side": turns_per_side,
                "arbiter_model": arbiter_model,
                "lr": getattr(config, "lr", None),
                "gamma": getattr(config, "gamma", None),
                "clip": getattr(config, "clip", None),
                "entropy_coef": getattr(config, "entropy_coef", None),
                "value_coef": getattr(config, "value_coef", None),
            },
        )
        assert hasattr(config, "clip"), "PPOConfig required"
        self.debate = DebateEnv(env, turns_per_side=turns_per_side)
        self.rl = {role: PPO(policy, config) for role, policy in self.policies.items()}

    def train(self):
        for ep in range(self.episodes):
            if self.curriculum is not None:
                self.env.dataset = [self.curriculum.sample(self._train_dataset)]

            state = self.debate.reset()
            for policy in self.policies.values():
                policy.reset_episode_cache()

            log = []
            total_tokens = 0
            done = False
            info: dict = {}

            while not done:
                role = self.debate.role
                policy = self.policies[role]
                action, payload, action_idx = policy.act(
                    state, allowed_actions=self.debate.current_allowed_actions()
                )
                step = {
                    "role": role,
                    "features": policy.last_features,
                    "mask": policy.last_mask,
                    "action_idx": action_idx,
                    "prob": policy.last_probs[action_idx],
                    "value": policy.get_value(state),
                    "action": action,
                    "entropy": policy.last_entropy,
                    "action_probs": policy.last_probs.tolist(),
                    "policy_type": policy.__class__.__name__,
                    "payload": payload,
                    "claim": state.claim,
                }

                state, reward, done, info = self.debate.step(action, payload)

                tokens = payload.get("tokens", 0) if isinstance(payload, dict) else 0
                total_tokens += tokens
                step["raw_reward"] = reward - self.token_penalty * tokens
                step["tokens"] = tokens
                step["llm_reward"] = info.get("llm_reward", 0)
                step["llm_scores"] = info.get("llm_scores", {})
                step["selected_ids"] = [e.id for e in state.selected_evidence]
                step["evidence_pool"] = [{"id": e.id, "text": e.text} for e in state.evidence_pool]
                log.append(step)

            final_raw = info["final_reward"]
            # Normalizer statistics see every step reward and the final reward once.
            for step in log:
                self.reward_rms.update([step["raw_reward"]])
                step["reward"] = float(self.reward_rms.normalize(step["raw_reward"]))
            self.reward_rms.update([final_raw])
            final = float(self.reward_rms.normalize(final_raw))

            for role in ROLES:
                [s for s in log if s["role"] == role][-1]["reward"] += final

            for role in ROLES:
                own = [s for s in log if s["role"] == role]
                self.rl[role].update([
                    (s["features"], s["mask"], s["action_idx"], s["prob"], s["reward"], s["value"])
                    for s in own
                ])

            claim_id = self.env.current_sample.get("id", self.env.current_sample.get("claim"))
            self._update_curriculum(claim_id, info.get("task_success"))

            total_reward = sum(s["reward"] for s in log) - final
            total_reward_raw = sum(s["raw_reward"] for s in log) + final_raw
            viz = [
                {
                    "step": i + 1,
                    "role": s["role"],
                    "action": s["action"],
                    "action_idx": s["action_idx"],
                    "reward": s["reward"],
                    "llm_reward": s["llm_reward"],
                    "entropy": s["entropy"],
                    "action_probs": s["action_probs"],
                    "policy_type": s["policy_type"],
                    "value_estimate": s["value"],
                    "llm_scores": s["llm_scores"],
                    "tokens": s["tokens"],
                    "action_names": ACTIONS,
                    "argument": s["payload"].get("argument", "") if isinstance(s["payload"], dict) else "",
                    "action_payload": s["payload"] if isinstance(s["payload"], dict) else {},
                    "selected_ids": s["selected_ids"],
                    "claim": s["claim"],
                    "evidence_pool": s["evidence_pool"],
                }
                for i, s in enumerate(log)
            ]
            metrics = self._build_episode_metrics(
                ep, total_reward, total_reward_raw, len(log), total_tokens, viz
            )
            metrics["entropy"] = float(np.mean([p.last_entropy for p in self.policies.values()]))

            if self.use_wandb:
                import wandb
                wandb.log(metrics, step=ep)

            self.tracker.log_episode(metrics)
            self.tracker.save_trajectory(ep, viz)
            print(f"Ep {ep} | Reward {total_reward:.3f}")

        for role, policy in self.policies.items():
            path = os.path.join(self.tracker.base_dir, f"policy_{role}.npz")
            policy.save(path)
            print(f"Checkpoint saved: {path}")
        self._finish()
