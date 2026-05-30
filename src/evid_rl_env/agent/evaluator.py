import numpy as np
from evid_rl_env.environment.actions import ACTIONS


class Evaluator:
    """Runs a fixed held-out eval split with no policy updates."""

    def __init__(self, env, policy, n_eval_episodes=20, reward_normalizer=None):
        self.env = env
        self.policy = policy
        self.n_eval_episodes = n_eval_episodes
        # RunningMeanStd instance or None. Used read-only — stats are never
        # updated here so training normalization stays uncontaminated.
        self.reward_normalizer = reward_normalizer

    def evaluate(self):
        print("[Eval] greedy mode active — running evaluation...")
        # AUDIT FIX: log the LLM seed so every eval run is auditable from logs;
        # the seed is stored on policy.llm.seed if the policy uses an LLMClient
        _llm_seed = getattr(getattr(self.policy, "llm", None), "seed", None)
        if _llm_seed is not None:
            print(f"[Eval] LLM seed: {_llm_seed}")
        # AUDIT FIX: lock the reward normalizer for the duration of this eval pass so
        # any accidental .update() call inside the loop raises immediately rather than
        # silently corrupting the training normalizer state
        if self.reward_normalizer is not None:
            self.reward_normalizer._locked = True

        rewards_raw = []
        steps_list = []
        action_counts = {a: 0 for a in ACTIONS}
        llm_score_totals = {"LCS": [], "ESS": [], "HRS": [], "COMP": []}

        try:
            for _ in range(self.n_eval_episodes):
                state = self.env.reset()
                done = False
                ep_reward = 0.0
                ep_steps = 0

                while not done:
                    action, payload, action_idx = self.policy.act(state, greedy=True)
                    next_state, reward, done, info = self.env.step(action, payload)
                    ep_reward += reward
                    ep_steps += 1
                    action_counts[action] = action_counts.get(action, 0) + 1

                    for k in llm_score_totals:
                        v = info.get("llm_scores", {}).get(k)
                        if v is not None:
                            llm_score_totals[k].append(v)

                    state = next_state

                rewards_raw.append(ep_reward)
                steps_list.append(ep_steps)
        finally:
            # AUDIT FIX: always unlock so the normalizer can be updated by training
            # even if an exception occurs mid-eval
            if self.reward_normalizer is not None:
                self.reward_normalizer._locked = False

        # Optionally normalize using training stats (read-only — no .update() calls)
        if self.reward_normalizer is not None:
            rewards_norm = [
                float(self.reward_normalizer.normalize(r)) for r in rewards_raw
            ]
        else:
            rewards_norm = rewards_raw

        total_actions = max(sum(action_counts.values()), 1)
        return {
            # Normalized (or raw if no normalizer) — primary signal for tracking
            "eval/mean_reward": float(np.mean(rewards_norm)),
            "eval/std_reward": float(np.std(rewards_norm)),
            # Raw always present so baselines and RL can be compared on the same scale
            "eval/mean_reward_raw": float(np.mean(rewards_raw)),
            "eval/std_reward_raw": float(np.std(rewards_raw)),
            "eval/mean_steps": float(np.mean(steps_list)),
            "eval/action_dist": {k: v / total_actions for k, v in action_counts.items()},
            "eval/llm_LCS": float(np.mean(llm_score_totals["LCS"])) if llm_score_totals["LCS"] else 0.0,
            "eval/llm_ESS": float(np.mean(llm_score_totals["ESS"])) if llm_score_totals["ESS"] else 0.0,
            "eval/llm_HRS": float(np.mean(llm_score_totals["HRS"])) if llm_score_totals["HRS"] else 0.0,
            "eval/llm_COMP": float(np.mean(llm_score_totals["COMP"])) if llm_score_totals["COMP"] else 0.0,
        }
