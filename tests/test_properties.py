import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from evid_rl_env.environment.curriculum import Curriculum
from evid_rl_env.environment.state import Evidence, State
from evid_rl_env.judge.llm_judge import LLMJudge
from evid_rl_env.judge.metrics import (
    compute_adversarial_contamination,
    compute_contradiction_acknowledgment,
    compute_f1,
    compute_precision,
    compute_recall,
)
from evid_rl_env.judge.reward import RewardFunction
from evid_rl_env.utils.running_stats import RunningMeanStd
from evid_rl_env.utils.smoothing import smooth_reward

LABELS = ["support", "contradict", "neutral", "adversarial"]
unit = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)


@st.composite
def pool_and_selection(draw):
    labels = draw(st.lists(st.sampled_from(LABELS), max_size=12))
    pool = [Evidence(id=i, text=f"doc {i}", label=lab) for i, lab in enumerate(labels)]
    selected = [e for e in pool if draw(st.booleans())]
    return pool, selected


@given(pool_and_selection())
def test_metrics_stay_in_unit_interval(data):
    pool, selected = data
    for value in (
        compute_precision(selected),
        compute_recall(selected, pool),
        compute_f1(selected, pool),
        compute_contradiction_acknowledgment(selected, pool),
        compute_adversarial_contamination(selected),
    ):
        assert 0.0 <= value <= 1.0


@given(pool_and_selection())
def test_f1_never_exceeds_larger_of_precision_and_recall(data):
    pool, selected = data
    p, r = compute_precision(selected), compute_recall(selected, pool)
    assert compute_f1(selected, pool) <= max(p, r) + 1e-12


@given(pool_and_selection())
def test_selecting_everything_acknowledges_every_contradiction(data):
    pool, _ = data
    assert compute_contradiction_acknowledgment(pool, pool) == 1.0
    assert compute_recall(pool, pool) == 1.0


@given(pool_and_selection(), unit, unit)
def test_final_reward_is_bounded(data, confidence, true_score):
    pool, selected = data
    state = State(claim="c", evidence_pool=pool, selected_evidence=selected)
    out = {"reasoning": "because", "confidence": confidence, "true_score": true_score}
    reward = RewardFunction().compute(state, out, llm_reward=0.5)
    assert -1.0 <= reward <= 1.0


@given(pool_and_selection(), unit, unit)
def test_final_reward_does_not_increase_with_calibration_error(data, confidence, true_score):
    pool, selected = data
    state = State(claim="c", evidence_pool=pool, selected_evidence=selected)
    fn = RewardFunction()
    calibrated = fn.compute(
        state, {"reasoning": "x", "confidence": true_score, "true_score": true_score}, 0.5
    )
    off = fn.compute(
        state, {"reasoning": "x", "confidence": confidence, "true_score": true_score}, 0.5
    )
    assert off <= calibrated + 1e-12


@given(*[unit] * 6)
def test_judge_reward_is_in_unit_interval(lcs, ess, grs, comp, bias, conf):
    scores = {"LCS": lcs, "ESS": ess, "GRS": grs, "COMP": comp, "BIAS": bias, "confidence": conf}
    assert 0.0 <= LLMJudge._scores_to_reward(scores) <= 1.0


@given(unit, unit, unit, unit, unit)
def test_zero_judge_confidence_gives_neutral_reward(lcs, ess, grs, comp, bias):
    scores = {"LCS": lcs, "ESS": ess, "GRS": grs, "COMP": comp, "BIAS": bias, "confidence": 0.0}
    assert LLMJudge._scores_to_reward(scores) == pytest.approx(0.5)


@given(st.lists(st.floats(-1e3, 1e3, allow_nan=False), min_size=1, max_size=50),
       st.integers(min_value=1, max_value=10))
def test_smoothed_reward_stays_within_raw_range(values, window):
    df = smooth_reward(pd.DataFrame({"reward": values}), window=window)
    assert df["reward_smooth"].min() >= min(values) - 1e-9
    assert df["reward_smooth"].max() <= max(values) + 1e-9


@settings(deadline=None)
@given(st.lists(st.floats(-1e3, 1e3, allow_nan=False), min_size=2, max_size=40))
def test_running_stats_match_numpy_for_one_batch(values):
    stats = RunningMeanStd(epsilon=1e-12)
    stats.update(values)
    assert stats.mean == pytest.approx(np.mean(values), abs=1e-6)
    assert stats.var >= 0.0


@given(st.lists(st.lists(unit, min_size=1, max_size=8), min_size=1, max_size=12),
       st.integers(min_value=1, max_value=6))
def test_curriculum_scores_are_positive_so_every_claim_is_sampleable(histories, n_claims):
    cur = Curriculum()
    dataset = [{"id": i, "claim": f"c{i}"} for i in range(n_claims)]
    for i, history in enumerate(histories):
        for perf in history:
            cur.record(i % n_claims, perf)
    assert all(cur.score(i) > 0 for i in range(n_claims))
    assert cur.sample(dataset) in dataset
