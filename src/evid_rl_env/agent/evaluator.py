import numpy as np
from evid_rl_env.environment.actions import ACTIONS


class Evaluator:
    """Runs a fixed held-out eval split with no policy updates."""

    def __init__(self, env, policy, n_eval_episodes=20):
        self.env = env
        self.policy = policy
        self.n_eval_episodes = n_eval_episodes

    def evaluate(self):
        rewards = []
        steps_list = []
        action_counts = {a: 0 for a in ACTIONS}
        llm_score_totals = {"LCS": [], "ESS": [], "HRS": [], "COMP": []}

        for _ in range(self.n_eval_episodes):
            state = self.env.reset()
            done = False
            ep_reward = 0.0
            ep_steps = 0

            while not done:
                action, payload, action_idx = self.policy.act(state)
                next_state, reward, done, info = self.env.step(action, payload)
                ep_reward += reward
                ep_steps += 1
                action_counts[action] = action_counts.get(action, 0) + 1

                for k in llm_score_totals:
                    v = info.get("llm_scores", {}).get(k)
                    if v is not None:
                        llm_score_totals[k].append(v)

                state = next_state

            rewards.append(ep_reward)
            steps_list.append(ep_steps)

        total_actions = max(sum(action_counts.values()), 1)
        return {
            "eval/mean_reward": float(np.mean(rewards)),
            "eval/std_reward": float(np.std(rewards)),
            "eval/mean_steps": float(np.mean(steps_list)),
            "eval/action_dist": {k: v / total_actions for k, v in action_counts.items()},
            "eval/llm_LCS": float(np.mean(llm_score_totals["LCS"])) if llm_score_totals["LCS"] else 0.0,
            "eval/llm_ESS": float(np.mean(llm_score_totals["ESS"])) if llm_score_totals["ESS"] else 0.0,
            "eval/llm_HRS": float(np.mean(llm_score_totals["HRS"])) if llm_score_totals["HRS"] else 0.0,
            "eval/llm_COMP": float(np.mean(llm_score_totals["COMP"])) if llm_score_totals["COMP"] else 0.0,
        }
