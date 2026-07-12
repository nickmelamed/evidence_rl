import pytest

from evid_rl_env.environment.state import Evidence, State
from evid_rl_env.judge.reward import RewardFunction

SUPPORT = Evidence(id=0, text="support", label="support")


def _state(selected=None, pool=None):
    pool = pool if pool is not None else [SUPPORT]
    selected = selected if selected is not None else [SUPPORT]
    return State(claim="claim", evidence_pool=pool, selected_evidence=selected)


def test_empty_reasoning_short_circuits_to_zero():
    reward_fn = RewardFunction()
    state = _state()
    assert reward_fn.compute(state, {"reasoning": "   "}, llm_reward=0.0) == 0.0


def test_llm_reward_value_applies_its_own_weight_directly():
    reward_fn = RewardFunction()
    state = _state()
    final_output = {"reasoning": "some reasoning", "confidence": 0.5, "true_score": 0.5}
    reward = reward_fn.compute(state, final_output, llm_reward=0.9)
    # 0.40*f1(1.0) + 0.20*ca(1.0) + 0.10*0.9 = 0.69, w_f1/w_contradiction NOT boosted
    assert reward == pytest.approx(0.69)


def test_uncertainty_penalty_reduces_reward():
    reward_fn = RewardFunction()
    state = _state()
    confident_wrong = {"reasoning": "r", "confidence": 1.0, "true_score": 0.0}
    confident_right = {"reasoning": "r", "confidence": 1.0, "true_score": 1.0}
    reward_wrong = reward_fn.compute(state, confident_wrong, llm_reward=0.0)
    reward_right = reward_fn.compute(state, confident_right, llm_reward=0.0)
    assert reward_wrong < reward_right


def test_overselection_penalty():
    # Pool == selected in both cases (precision=recall=f1=1.0 either way) so the
    # penalty term is the *only* thing that differs between over/within budget.
    reward_fn = RewardFunction(evidence_budget=2)
    final_output = {"reasoning": "r", "confidence": 0.5, "true_score": 0.5}

    within_budget = [Evidence(id=i, text=f"e{i}", label="support") for i in range(2)]
    within_state = _state(selected=within_budget, pool=within_budget)

    over_budget = [Evidence(id=i, text=f"e{i}", label="support") for i in range(5)]
    over_state = _state(selected=over_budget, pool=over_budget)  # penalty = 0.04*(5-2) = 0.12

    reward_within = reward_fn.compute(within_state, final_output, llm_reward=0.5)
    reward_over = reward_fn.compute(over_state, final_output, llm_reward=0.5)
    assert reward_over == pytest.approx(reward_within - 0.12)


def test_reward_is_clipped_to_unit_interval():
    reward_fn = RewardFunction()
    state = _state(selected=[], pool=[SUPPORT])
    final_output = {"reasoning": "r", "confidence": 1.0, "true_score": 0.0}
    reward = reward_fn.compute(state, final_output, llm_reward=0.0)
    assert 0.0 <= reward <= 1.0
