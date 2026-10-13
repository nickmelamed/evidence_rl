import numpy as np
import pytest

from evid_rl_env.agent.debate_evaluator import DebateEvaluator
from evid_rl_env.agent.evaluator import Evaluator
from evid_rl_env.environment.actions import Actions

DATASET = [
    {"claim": f"Held-out claim {i}.", "search_query": f"held-out claim {i}", "label": i % 2}
    for i in range(6)
]


def _evaluator(fake_policy, make_env, n=3, seed=0):
    return DebateEvaluator(
        make_env(dataset=DATASET, seed=seed),
        fake_policy(state_dim=4, seed=1),
        fake_policy(state_dim=4, seed=2),
        turns_per_side=5,
        n_eval_episodes=n,
    )


def test_the_report_has_the_keys_the_single_agent_evaluator_reports(fake_policy, make_env):
    np.random.seed(0)
    result = _evaluator(fake_policy, make_env).evaluate()
    for key in ("eval/mean_reward_raw", "eval/std_reward_raw", "eval/mean_steps",
                "eval/action_dist", "eval/llm_LCS", "eval/llm_BIAS"):
        assert key in result
    assert result["eval/mean_steps"] == 10


def test_an_episode_reward_is_every_step_plus_the_final_reward(fake_policy, make_env):
    np.random.seed(0)
    evaluator = _evaluator(fake_policy, make_env, n=1)
    totals = []
    original = DebateEvaluator._play_episode

    def recording(self, debate, *args, **kwargs):
        step = debate.step
        seen = []

        def wrapped(action, payload):
            result = step(action, payload)
            seen.append(result[1])
            if result[2]:
                seen.append(result[3]["final_reward"])
            return result

        debate.step = wrapped
        out = original(self, debate, *args, **kwargs)
        totals.append(sum(seen))
        return out

    DebateEvaluator._play_episode = recording
    try:
        result = evaluator.evaluate()
    finally:
        DebateEvaluator._play_episode = original
    assert result["eval/mean_reward_raw"] == pytest.approx(totals[0])


def test_it_plays_the_same_claims_in_the_same_order_as_the_single_agent_evaluator(
    fake_policy, make_env
):
    def claims_seen(env, run):
        seen = []
        original = env.reset
        env.reset = lambda: (original(), seen.append(env.current_sample["claim"]))[0]
        run()
        return seen

    np.random.seed(0)
    debate_env = make_env(dataset=DATASET, seed=7)
    debate = DebateEvaluator(debate_env, fake_policy(state_dim=4, seed=1),
                             fake_policy(state_dim=4, seed=2), n_eval_episodes=4)
    single_env = make_env(dataset=DATASET, seed=7)
    single = Evaluator(single_env, fake_policy(state_dim=4, seed=1), n_eval_episodes=4)

    assert claims_seen(debate_env, debate.evaluate) == claims_seen(single_env, single.evaluate)


def test_evaluation_is_deterministic_and_does_not_change_the_weights(fake_policy, make_env):
    def run():
        np.random.seed(5)
        evaluator = _evaluator(fake_policy, make_env, seed=3)
        before = (evaluator.support_policy.actor_params.copy(),
                  evaluator.contradict_policy.actor_params.copy())
        result = evaluator.evaluate()
        np.testing.assert_array_equal(before[0], evaluator.support_policy.actor_params)
        np.testing.assert_array_equal(before[1], evaluator.contradict_policy.actor_params)
        return result

    assert run() == run()


def test_each_policy_stays_inside_its_role(fake_policy, make_env):
    np.random.seed(0)
    evaluator = _evaluator(fake_policy, make_env)
    taken = {"support": [], "contradict": []}
    for role, policy in (("support", evaluator.support_policy),
                         ("contradict", evaluator.contradict_policy)):
        original = policy.act

        def recording(state, role=role, original=original, **kwargs):
            out = original(state, **kwargs)
            taken[role].append(out[0])
            return out

        policy.act = recording
    evaluator.evaluate()
    assert Actions.CONTRADICT not in taken["support"]
    assert Actions.SUPPORT not in taken["contradict"]
    assert Actions.FINALIZE not in taken["support"] + taken["contradict"]
