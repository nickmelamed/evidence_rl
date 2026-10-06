import random

from evid_rl_env.agent.base_trainer import build_standard_baselines

DATASET = [{"id": i, "claim": f"claim {i}", "search_query": f"q{i}"} for i in range(30)]


def _claims(env, n=12):
    return [env.reset() and env.current_sample["id"] for _ in range(n)]


def test_same_seed_gives_same_claim_sequence(make_env):
    assert _claims(make_env(dataset=DATASET, seed=3)) == _claims(make_env(dataset=DATASET, seed=3))


def test_different_seeds_give_different_claim_sequences(make_env):
    assert _claims(make_env(dataset=DATASET, seed=3)) != _claims(make_env(dataset=DATASET, seed=4))


def test_claim_sequence_ignores_draws_from_the_global_rng(make_env):
    env_a = make_env(dataset=DATASET, seed=5)
    random.seed(1)
    first = []
    for _ in range(12):
        random.random()
        first.append(env_a.reset() and env_a.current_sample["id"])

    env_b = make_env(dataset=DATASET, seed=5)
    random.seed(999)
    second = _claims(env_b)

    assert first == second


def test_reseed_restarts_the_sequence(make_env):
    env = make_env(dataset=DATASET, seed=7)
    first = _claims(env)
    env.reseed()
    assert _claims(env) == first


def test_evaluator_reseeds_the_env_every_round(make_env, fake_policy):
    from evid_rl_env.agent.evaluator import Evaluator

    env = make_env(dataset=DATASET, seed=9)
    evaluator = Evaluator(env, fake_policy(), n_eval_episodes=3, reward_normalizer=None)
    seen = []
    original_reset = env.reset

    def spy():
        state = original_reset()
        seen.append(env.current_sample["id"])
        return state

    env.reset = spy
    evaluator.evaluate()
    first_round = list(seen)
    seen.clear()
    evaluator.evaluate()
    assert seen == first_round


def test_build_standard_baselines_gives_every_baseline_the_eval_seed(mock_llm):
    baselines = build_standard_baselines(DATASET, DATASET, mock_llm(response="{}"), seed=11)
    assert {b.seed for b in baselines.values()} == {11}
