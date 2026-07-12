"""
Unit tests for EnsembleJudge (judge/ensemble_judge.py): median aggregation
across members, measured-disagreement confidence, empty-reasoning
short-circuit, and interface compatibility with LLMJudge — all against
mocked LLM clients, no real model loads (same style as
tests/test_llm_judge_smoke.py).
"""

import pytest

from evid_rl_env.environment.state import Evidence
from evid_rl_env.judge.ensemble_judge import EnsembleJudge
from evid_rl_env.judge.llm_judge import LLMJudge

CLAIM = "The Earth orbits the Sun."
REASONING = "Well-established heliocentric astronomy confirms this."
EVIDENCE = [Evidence(id=0, text="Kepler's laws describe planetary orbits around the Sun.", label="support")]


class _MockLLM:
    def __init__(self, response: str):
        self._response = response
        self.calls = 0

    def generate(self, prompt):
        self.calls += 1
        return self._response, len(self._response.split())

    def generate_structured(self, prompt, temperature=None):
        return self.generate(prompt)


class _RaisingLLM:
    """Used to prove a code path never actually calls the model."""

    def generate(self, prompt):
        raise AssertionError("should not be called")

    def generate_structured(self, prompt, temperature=None):
        raise AssertionError("should not be called")


def _judge(response: str, tmp_path, name: str) -> LLMJudge:
    return LLMJudge(_MockLLM(response), cache_path=str(tmp_path / f"{name}.sqlite3"))


def _scores_json(v: float, confidence: float = 0.9) -> str:
    return (
        f'{{"LCS": {v}, "ESS": {v}, "GRS": {1 - v}, "COMP": {v}, "BIAS": {1 - v}, '
        f'"confidence": {confidence}}}'
    )


def test_requires_at_least_two_judges(tmp_path):
    with pytest.raises(ValueError):
        EnsembleJudge([_judge(_scores_json(0.8), tmp_path, "a")])


def test_median_aggregation_two_members(tmp_path):
    judge_a = _judge(_scores_json(0.8), tmp_path, "a")
    judge_b = _judge(_scores_json(0.6), tmp_path, "b")

    ensemble = EnsembleJudge([judge_a, judge_b])
    reward, scores = ensemble.compute_reward(CLAIM, REASONING, EVIDENCE)

    assert scores["LCS"] == pytest.approx(0.7)
    assert scores["ESS"] == pytest.approx(0.7)
    assert scores["COMP"] == pytest.approx(0.7)


def test_full_agreement_yields_confidence_one(tmp_path):
    judge_a = _judge(_scores_json(0.8), tmp_path, "a")
    judge_b = _judge(_scores_json(0.8), tmp_path, "b")

    ensemble = EnsembleJudge([judge_a, judge_b])
    _, scores = ensemble.compute_reward(CLAIM, REASONING, EVIDENCE)

    assert scores["confidence"] == pytest.approx(1.0)


def test_maximal_disagreement_yields_confidence_zero(tmp_path):
    judge_a = _judge(_scores_json(1.0), tmp_path, "a")
    judge_b = _judge(_scores_json(0.0), tmp_path, "b")

    ensemble = EnsembleJudge([judge_a, judge_b])
    _, scores = ensemble.compute_reward(CLAIM, REASONING, EVIDENCE)

    assert scores["confidence"] == pytest.approx(0.0)


def test_reward_matches_llmjudge_formula_on_aggregated_scores(tmp_path):
    judge_a = _judge(_scores_json(0.8), tmp_path, "a")
    judge_b = _judge(_scores_json(0.6), tmp_path, "b")

    ensemble = EnsembleJudge([judge_a, judge_b])
    reward, scores = ensemble.compute_reward(CLAIM, REASONING, EVIDENCE)

    assert reward == pytest.approx(LLMJudge._scores_to_reward(scores))


def test_empty_reasoning_short_circuits_without_calling_members(tmp_path):
    judge_a = LLMJudge(_RaisingLLM(), cache_path=str(tmp_path / "a.sqlite3"))
    judge_b = LLMJudge(_RaisingLLM(), cache_path=str(tmp_path / "b.sqlite3"))

    ensemble = EnsembleJudge([judge_a, judge_b])
    reward, scores = ensemble.compute_reward(CLAIM, "", EVIDENCE)

    assert reward == 0.0
    assert scores["LCS"] == 0.0


def test_scores_dict_has_all_dimensions_and_confidence(tmp_path):
    judge_a = _judge(_scores_json(0.8), tmp_path, "a")
    judge_b = _judge(_scores_json(0.6), tmp_path, "b")

    ensemble = EnsembleJudge([judge_a, judge_b])
    _, scores = ensemble.compute_reward(CLAIM, REASONING, EVIDENCE)

    for key in ("LCS", "ESS", "GRS", "COMP", "BIAS", "confidence"):
        assert key in scores
