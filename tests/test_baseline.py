import json
import logging
import random

import pytest

from evid_rl_env.agent.baseline import (
    N_ACTIONS,
    BestOfNBaseline,
    FewShotLLMBaseline,
    GreedyLLMBaseline,
    ImitationBaseline,
    MajorityBaseline,
    RandomBaseline,
)
from evid_rl_env.environment.state import Evidence, State

GOOD_SCORES_JSON = (
    '{"LCS": 0.8, "ESS": 0.8, "GRS": 0.2, "COMP": 0.8, "BIAS": 0.2, "confidence": 0.9}'
)

EVAL_DATASET = [{"claim": "Baselines claim.", "search_query": "baselines claim"}]


@pytest.fixture
def patch_baseline_env(monkeypatch, make_llm_judge, fake_evidence_labeler):
    """RandomBaseline/MajorityBaseline/etc. each construct their own ClaimEnv
    with no way to inject a judge from outside — patch the name baseline.py
    itself uses so tests never need a real generation/judge model."""
    from evid_rl_env.environment.environment import ClaimEnv as RealClaimEnv

    def _patch(mock_response: str = GOOD_SCORES_JSON, **mock_kwargs):
        judge = make_llm_judge(mock_response, **mock_kwargs)

        def _fake_claim_env(dataset, *args, **kwargs):
            return RealClaimEnv(dataset, llm_judge=judge, evidence_labeler=fake_evidence_labeler())

        monkeypatch.setattr("evid_rl_env.agent.baseline.ClaimEnv", _fake_claim_env)
        return judge

    return _patch


def _state(claim="Test claim"):
    return State(claim=claim, evidence_pool=[Evidence(id=0, text="some evidence", label="neutral")])


# ---------------------------------------------------------------------------
# Random / Majority
# ---------------------------------------------------------------------------

def test_random_baseline_is_deterministic_under_a_seeded_rng(patch_baseline_env):
    patch_baseline_env()
    baseline = RandomBaseline(EVAL_DATASET)

    random.seed(123)
    result_a = baseline.run(3)
    random.seed(123)
    result_b = baseline.run(3)

    assert result_a == result_b


def test_majority_baseline_caches_and_reuses_the_majority_action(patch_baseline_env):
    patch_baseline_env()
    baseline = MajorityBaseline(EVAL_DATASET)
    assert baseline._majority_idx is None

    random.seed(0)
    baseline.run(2)
    assert baseline._majority_idx is not None
    cached = baseline._majority_idx

    baseline.run(2)  # second run() must reuse the cached majority, not recompute
    assert baseline._majority_idx == cached


# ---------------------------------------------------------------------------
# GreedyLLMBaseline
# ---------------------------------------------------------------------------

def test_greedy_llm_baseline_parses_a_valid_action_index(mock_llm):
    baseline = GreedyLLMBaseline(EVAL_DATASET, mock_llm(response="2"))
    idx = baseline._action_fn(_state())
    assert idx == 2


def test_greedy_llm_baseline_falls_back_to_random_and_logs_on_bad_response(mock_llm, caplog):
    baseline = GreedyLLMBaseline(EVAL_DATASET, mock_llm(response="not a number"))
    with caplog.at_level(logging.WARNING):
        idx = baseline._action_fn(_state())
    assert 0 <= idx < N_ACTIONS
    assert any("could not be parsed" in r.message for r in caplog.records)


def test_greedy_llm_baseline_falls_back_to_random_and_logs_on_exception(mock_llm, caplog):
    baseline = GreedyLLMBaseline(EVAL_DATASET, mock_llm(raise_exc=RuntimeError("boom")))
    with caplog.at_level(logging.WARNING):
        idx = baseline._action_fn(_state())
    assert 0 <= idx < N_ACTIONS
    assert any("LLM call failed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# FewShotLLMBaseline
# ---------------------------------------------------------------------------

def test_fewshot_baseline_parses_a_valid_action_index(mock_llm, patch_baseline_env):
    patch_baseline_env()
    baseline = FewShotLLMBaseline(EVAL_DATASET, EVAL_DATASET, mock_llm(response="4"), k=1)
    idx = baseline._action_fn(_state(), k=1)
    assert idx == 4


def test_fewshot_baseline_similarity_mode_is_selectable(mock_llm, patch_baseline_env):
    patch_baseline_env()
    baseline = FewShotLLMBaseline(
        EVAL_DATASET, EVAL_DATASET, mock_llm(response="1"), k=1, selection_mode="similarity"
    )
    assert baseline.selection_mode == "similarity"
    idx = baseline._action_fn(_state(), k=1)
    assert idx == 1


# ---------------------------------------------------------------------------
# BestOfNBaseline
# ---------------------------------------------------------------------------

def test_best_of_n_suggest_action_falls_back_to_random_and_logs_on_exception(mock_llm, caplog):
    """Regression test: _suggest_action used to swallow exceptions silently,
    unlike GreedyLLMBaseline/FewShotLLMBaseline which both log a warning."""
    baseline = BestOfNBaseline(EVAL_DATASET, mock_llm(raise_exc=RuntimeError("boom")), n=3)
    with caplog.at_level(logging.WARNING):
        idx = baseline._suggest_action(_state())
    assert 0 <= idx < N_ACTIONS
    assert any("LLM call failed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# ImitationBaseline
# ---------------------------------------------------------------------------

def _write_trajectories(path, records):
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def test_imitation_baseline_exact_match(tmp_path):
    from evid_rl_env.agent.baseline import _state_summary

    state = _state(claim="Imitation claim")
    obs = _state_summary(state)

    traj_path = tmp_path / "trajectories.jsonl"
    _write_trajectories(traj_path, [{
        "claim": "Imitation claim",
        "annotator_model": "mock",
        "mode": "llm_annotator",
        "steps": [{"obs": obs, "action": 3}],
    }])

    baseline = ImitationBaseline(EVAL_DATASET, str(traj_path))
    assert baseline._action_fn(state) == 3


def test_imitation_baseline_falls_back_to_random_for_unknown_claim(tmp_path):
    traj_path = tmp_path / "trajectories.jsonl"
    _write_trajectories(traj_path, [{
        "claim": "Some other claim",
        "annotator_model": "mock",
        "mode": "llm_annotator",
        "steps": [{"obs": "irrelevant", "action": 0}],
    }])

    baseline = ImitationBaseline(EVAL_DATASET, str(traj_path))
    idx = baseline._action_fn(_state(claim="Totally unseen claim"))
    assert 0 <= idx < N_ACTIONS
