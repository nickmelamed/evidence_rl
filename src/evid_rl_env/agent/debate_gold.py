from evid_rl_env.agent.gold_evaluator import GoldEvaluator
from evid_rl_env.environment.debate_env import DebateEnv


class _PairEnv:
    """Present a debate as the single-agent env that ``GoldEvaluator`` expects."""

    def __init__(self, debate: DebateEnv):
        self.debate = debate

    @property
    def current_sample(self):
        return self.debate.env.current_sample

    def reseed(self):
        self.debate.env.reseed()

    def reset(self):
        return self.debate.reset()

    def step(self, action, payload):
        return self.debate.step(action, payload)


class _PairPolicy:
    """Route each ``act`` call to the role whose turn it is."""

    def __init__(self, debate: DebateEnv, support_policy, contradict_policy):
        self.debate = debate
        self.policies = {"support": support_policy, "contradict": contradict_policy}

    def reset_episode_cache(self):
        for policy in self.policies.values():
            policy.reset_episode_cache()

    def act(self, state, greedy=False):
        policy = self.policies[self.debate.role]
        return policy.act(
            state, greedy=greedy, allowed_actions=self.debate.current_allowed_actions()
        )


def build_debate_gold_evaluator(env, support_policy, contradict_policy, gold_judges,
                                n_episodes, turns_per_side, max_tokens) -> GoldEvaluator:
    """Audit the arbiter that scores a debate against held-out gold judges.

    ``env`` is a ``ClaimEnv`` whose judge is the training arbiter, so the proxy
    side is the arbiter's final score. The gold judges re-score the same full
    transcript. The rollout samples stochastically, like ``GoldEvaluator`` does.
    """
    debate = DebateEnv(env, turns_per_side=turns_per_side, max_tokens=max_tokens)
    return GoldEvaluator(
        _PairEnv(debate), _PairPolicy(debate, support_policy, contradict_policy),
        gold_judges, n_episodes=n_episodes,
    )
