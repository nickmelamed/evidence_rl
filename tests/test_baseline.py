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


# Random / Majority


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



# GreedyLLMBaseline


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


def test_shared_example_bank_builds_once_across_k3_and_k5(mock_llm, monkeypatch):
    """The actual fix: fewshot_k3 and fewshot_k5 sharing one
    _SharedFewShotExamples must only run the expensive random-policy sweep
    once between them, not once each."""
    import evid_rl_env.agent.baseline as baseline_module

    call_count = {"n": 0}

    def _fake_collect(train_dataset):
        call_count["n"] += 1
        return [{"observation": "obs", "action_idx": 0, "claim": "c"}]

    monkeypatch.setattr(baseline_module, "_collect_fewshot_examples", _fake_collect)

    bank = baseline_module._SharedFewShotExamples(EVAL_DATASET)
    k3 = FewShotLLMBaseline(EVAL_DATASET, EVAL_DATASET, mock_llm(response="0"), k=3, example_bank=bank)
    k5 = FewShotLLMBaseline(EVAL_DATASET, EVAL_DATASET, mock_llm(response="0"), k=5, example_bank=bank)

    k3._action_fn(_state(), k=3)
    k5._action_fn(_state(), k=5)

    assert call_count["n"] == 1


def test_without_shared_bank_each_instance_builds_independently(mock_llm, monkeypatch):
    """Regression: constructing FewShotLLMBaseline without an example_bank
    (e.g. standalone/test usage) must keep today's behavior — each instance
    builds and owns its own bank."""
    import evid_rl_env.agent.baseline as baseline_module

    call_count = {"n": 0}

    def _fake_collect(train_dataset):
        call_count["n"] += 1
        return [{"observation": "obs", "action_idx": 0, "claim": "c"}]

    monkeypatch.setattr(baseline_module, "_collect_fewshot_examples", _fake_collect)

    k3 = FewShotLLMBaseline(EVAL_DATASET, EVAL_DATASET, mock_llm(response="0"), k=3)
    k5 = FewShotLLMBaseline(EVAL_DATASET, EVAL_DATASET, mock_llm(response="0"), k=5)

    k3._action_fn(_state(), k=3)
    k5._action_fn(_state(), k=5)

    assert call_count["n"] == 2


def test_build_standard_baselines_shares_one_bank_between_k3_and_k5(mock_llm):
    from evid_rl_env.agent.base_trainer import build_standard_baselines

    baselines = build_standard_baselines(EVAL_DATASET, EVAL_DATASET, mock_llm(response="0"))

    assert baselines["fewshot_k3"]._example_bank is baselines["fewshot_k5"]._example_bank



# BestOfNBaseline


def test_best_of_n_suggest_action_falls_back_to_random_and_logs_on_exception(mock_llm, caplog):
    """Regression test: _suggest_action used to swallow exceptions silently,
    unlike GreedyLLMBaseline/FewShotLLMBaseline which both log a warning."""
    baseline = BestOfNBaseline(EVAL_DATASET, mock_llm(raise_exc=RuntimeError("boom")), n=3)
    with caplog.at_level(logging.WARNING):
        idx = baseline._suggest_action(_state())
    assert 0 <= idx < N_ACTIONS
    assert any("LLM call failed" in r.message for r in caplog.records)


def test_best_of_n_falls_back_to_sequential_calls_without_batched_method(mock_llm):
    """A client without generate_structured_n (e.g. this codebase's own
    duck-typed mock) must keep using n sequential _suggest_action() calls —
    today's exact behavior, unchanged."""
    baseline = BestOfNBaseline(EVAL_DATASET, mock_llm(response="2"), n=3)
    candidates = baseline._suggest_actions_batch(_state(), 3)
    assert candidates == [2, 2, 2]


def test_best_of_n_uses_batched_call_when_client_supports_it():
    """The actual fix: a client with generate_structured_n must be called
    once with n, not n times."""
    call_log = []

    class _BatchingLLM:
        model_name = "batching-mock"

        def generate_structured_n(self, prompt, n, temperature=0.1):
            call_log.append(n)
            return [(str(i % N_ACTIONS), 1) for i in range(n)]

    baseline = BestOfNBaseline(EVAL_DATASET, _BatchingLLM(), n=4)
    candidates = baseline._suggest_actions_batch(_state(), 4)

    assert call_log == [4]  # one call, not four
    assert candidates == [0, 1, 2, 3]


def test_best_of_n_batched_call_failure_falls_back_to_random_for_all(caplog):
    class _FailingBatchingLLM:
        model_name = "failing-batching-mock"

        def generate_structured_n(self, prompt, n, temperature=0.1):
            raise RuntimeError("boom")

    baseline = BestOfNBaseline(EVAL_DATASET, _FailingBatchingLLM(), n=3)
    with caplog.at_level(logging.WARNING):
        candidates = baseline._suggest_actions_batch(_state(), 3)

    assert len(candidates) == 3
    assert all(0 <= idx < N_ACTIONS for idx in candidates)
    assert any("batched LLM call failed" in r.message for r in caplog.records)



# ImitationBaseline


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
