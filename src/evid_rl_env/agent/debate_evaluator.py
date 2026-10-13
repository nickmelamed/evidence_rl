import numpy as np

from evid_rl_env.environment.actions import ACTIONS
from evid_rl_env.environment.debate_env import DebateEnv

_LLM_DIMENSIONS = ("LCS", "ESS", "GRS", "COMP", "BIAS")


class DebateEvaluator:
    """Play a trained support and contradict pair on a held-out split with no updates.

    The env should be the same single-judge ``ClaimEnv`` that ``evid-eval`` uses, so
    the pair is scored the same way as every other method. An episode's reward is
    the sum of both roles' step rewards plus the final reward, which makes it the
    reward of the pair acting as one agent. It reseeds the env, so the claims and
    their order match the single-agent evaluator's.
    """

    def __init__(self, env, support_policy, contradict_policy, turns_per_side=5,
                 n_eval_episodes=20, max_tokens=None):
        self.env = env
        self.support_policy = support_policy
        self.contradict_policy = contradict_policy
        self.turns_per_side = turns_per_side
        self.n_eval_episodes = n_eval_episodes
        self.max_tokens = max_tokens

    def evaluate(self) -> dict:
        print("[Eval] greedy mode active, running debate evaluation...")
        if hasattr(self.env, "reseed"):
            self.env.reseed()
        debate = DebateEnv(
            self.env, turns_per_side=self.turns_per_side, max_tokens=self.max_tokens
        )

        rewards = []
        steps_list = []
        action_counts = {a: 0 for a in ACTIONS}
        llm_scores: dict[str, list] = {k: [] for k in _LLM_DIMENSIONS}

        for _ in range(self.n_eval_episodes):
            ep_reward, ep_steps = self._play_episode(debate, action_counts, llm_scores)
            rewards.append(ep_reward)
            steps_list.append(ep_steps)

        total_actions = max(sum(action_counts.values()), 1)
        report = {
            "eval/mean_reward_raw": float(np.mean(rewards)),
            "eval/std_reward_raw": float(np.std(rewards)),
            "eval/mean_steps": float(np.mean(steps_list)),
            "eval/action_dist": {k: v / total_actions for k, v in action_counts.items()},
        }
        for k in _LLM_DIMENSIONS:
            report[f"eval/llm_{k}"] = float(np.mean(llm_scores[k])) if llm_scores[k] else 0.0
        return report

    def _play_episode(self, debate, action_counts, llm_scores):
        policies = {"support": self.support_policy, "contradict": self.contradict_policy}
        for policy in policies.values():
            policy.reset_episode_cache()
        state = debate.reset()
        done = False
        ep_reward = 0.0
        ep_steps = 0

        while not done:
            policy = policies[debate.role]
            action, payload, _ = policy.act(
                state, greedy=True, allowed_actions=debate.current_allowed_actions()
            )
            state, reward, done, info = debate.step(action, payload)
            ep_reward += reward
            ep_steps += 1
            action_counts[action] += 1
            for k in _LLM_DIMENSIONS:
                v = info.get("llm_scores", {}).get(k)
                if v is not None:
                    llm_scores[k].append(v)

        return ep_reward + info["final_reward"], ep_steps
