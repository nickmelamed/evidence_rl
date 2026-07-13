"""
Unit tests for EscalatingJudge (judge/escalating_judge.py): the three
escalation triggers (low confidence, high GRS, high BIAS, adversarial
evidence), the empty-reasoning short-circuit, and the "escalated" flag —
all against mocked LLM clients, no real model loads.
"""

import pytest

from evid_rl_env.environment.state import Evidence
from evid_rl_env.judge.escalating_judge import EscalatingJudge
from evid_rl_env.judge.llm_judge import LLMJudge

CLAIM = "The Earth orbits the Sun."
REASONING = "Well-established heliocentric astronomy confirms this."
CLEAN_EVIDENCE = [Evidence(id=0, text="Kepler's laws describe planetary orbits around the Sun.", label="support")]
ADVERSARIAL_EVIDENCE = [
    Evidence(id=0, text="Kepler's laws describe planetary orbits around the Sun.", label="support"),
    Evidence(id=1, text="A dubious blog post claims otherwise.", label="adversarial"),
]


class _MockLLM:
    def __init__(self, response: str):
        self._response = response

    def generate(self, prompt):
        return self._response, len(self._response.split())

    def generate_structured(self, prompt, temperature=None):
        return self.generate(prompt)


class _RaisingEscalatedJudge:
    """Used to prove escalation never happened."""

    def compute_reward(self, claim, reasoning, evidence):
        raise AssertionError("escalated_judge should not have been called")


class _StubEscalatedJudge:
    def __init__(self, reward, scores):
        self._reward = reward
        self._scores = scores

    def compute_reward(self, claim, reasoning, evidence):
        return self._reward, dict(self._scores)


def _cheap_judge(tmp_path, lcs=0.9, ess=0.9, grs=0.1, comp=0.9, bias=0.1, confidence=0.9) -> LLMJudge:
    response = (
        f'{{"LCS": {lcs}, "ESS": {ess}, "GRS": {grs}, "COMP": {comp}, '
        f'"BIAS": {bias}, "confidence": {confidence}}}'
    )
    return LLMJudge(_MockLLM(response), cache_path=str(tmp_path / "cheap.sqlite3"))


def test_no_escalation_when_confident_grounded_and_clean(tmp_path):
    cheap = _cheap_judge(tmp_path, grs=0.1, bias=0.1, confidence=0.9)
    escalating = EscalatingJudge(cheap, _RaisingEscalatedJudge())

    reward, scores = escalating.compute_reward(CLAIM, REASONING, CLEAN_EVIDENCE)

    assert scores["escalated"] is False
    assert reward == pytest.approx(LLMJudge._scores_to_reward(
        {"LCS": 0.9, "ESS": 0.9, "GRS": 0.1, "COMP": 0.9, "BIAS": 0.1, "confidence": 0.9}
    ))


def test_escalates_on_low_confidence(tmp_path):
    cheap = _cheap_judge(tmp_path, confidence=0.3)
    escalated = _StubEscalatedJudge(0.7, {"LCS": 0.7, "ESS": 0.7, "GRS": 0.2, "COMP": 0.7, "BIAS": 0.2, "confidence": 0.8})
    escalating = EscalatingJudge(cheap, escalated)

    reward, scores = escalating.compute_reward(CLAIM, REASONING, CLEAN_EVIDENCE)

    assert scores["escalated"] is True
    assert reward == 0.7


def test_escalates_on_high_grs(tmp_path):
    cheap = _cheap_judge(tmp_path, grs=0.9, confidence=0.9)
    escalated = _StubEscalatedJudge(0.4, {"LCS": 0.4, "ESS": 0.4, "GRS": 0.4, "COMP": 0.4, "BIAS": 0.4, "confidence": 0.6})
    escalating = EscalatingJudge(cheap, escalated)

    _, scores = escalating.compute_reward(CLAIM, REASONING, CLEAN_EVIDENCE)

    assert scores["escalated"] is True


def test_escalates_on_high_bias(tmp_path):
    cheap = _cheap_judge(tmp_path, bias=0.9, confidence=0.9)
    escalated = _StubEscalatedJudge(0.4, {"LCS": 0.4, "ESS": 0.4, "GRS": 0.4, "COMP": 0.4, "BIAS": 0.4, "confidence": 0.6})
    escalating = EscalatingJudge(cheap, escalated)

    _, scores = escalating.compute_reward(CLAIM, REASONING, CLEAN_EVIDENCE)

    assert scores["escalated"] is True


def test_escalates_on_adversarial_evidence_even_with_good_scores(tmp_path):
    cheap = _cheap_judge(tmp_path, grs=0.1, bias=0.1, confidence=0.95)
    escalated = _StubEscalatedJudge(0.5, {"LCS": 0.5, "ESS": 0.5, "GRS": 0.5, "COMP": 0.5, "BIAS": 0.5, "confidence": 0.5})
    escalating = EscalatingJudge(cheap, escalated)

    _, scores = escalating.compute_reward(CLAIM, REASONING, ADVERSARIAL_EVIDENCE)

    assert scores["escalated"] is True


def test_empty_reasoning_short_circuits_without_calling_either_tier(tmp_path):
    class _RaisingLLM:
        def generate(self, prompt):
            raise AssertionError("should not be called")

        def generate_structured(self, prompt, temperature=None):
            raise AssertionError("should not be called")

    cheap = LLMJudge(_RaisingLLM(), cache_path=str(tmp_path / "cheap.sqlite3"))
    escalating = EscalatingJudge(cheap, _RaisingEscalatedJudge())

    reward, scores = escalating.compute_reward(CLAIM, "", CLEAN_EVIDENCE)

    assert reward == 0.0
    assert scores["escalated"] is False
    assert scores["LCS"] == 0.0


def test_escalated_flag_present_on_both_paths(tmp_path):
    cheap = _cheap_judge(tmp_path)
    escalating = EscalatingJudge(cheap, _RaisingEscalatedJudge())
    _, scores = escalating.compute_reward(CLAIM, REASONING, CLEAN_EVIDENCE)
    assert "escalated" in scores

    cheap2 = _cheap_judge(tmp_path, confidence=0.1)
    escalated = _StubEscalatedJudge(0.5, {"LCS": 0.5, "ESS": 0.5, "GRS": 0.5, "COMP": 0.5, "BIAS": 0.5, "confidence": 0.5})
    escalating2 = EscalatingJudge(cheap2, escalated)
    _, scores2 = escalating2.compute_reward(CLAIM, REASONING, CLEAN_EVIDENCE)
    assert "escalated" in scores2


def test_shared_instance_with_ensemble_member_avoids_redundant_call(tmp_path):
    """The actual payoff of EnsembleJudge/build_ensemble_judge's `reuse`
    wiring in environment.py: when tier-1 and an ensemble member are the
    *same* LLMJudge instance, escalating doesn't re-query that model — the
    member's get_scores() hits the cache tier-1's own call already wrote,
    moments earlier, for the exact same (claim, reasoning, evidence)."""
    from evid_rl_env.judge.ensemble_judge import EnsembleJudge

    class _CountingLLM:
        def __init__(self, response):
            self._response = response
            self.calls = 0

        def generate(self, prompt):
            self.calls += 1
            return self._response, len(self._response.split())

        def generate_structured(self, prompt, temperature=None):
            return self.generate(prompt)

    shared_response = '{"LCS": 0.8, "ESS": 0.8, "GRS": 0.2, "COMP": 0.8, "BIAS": 0.2, "confidence": 0.9}'
    shared_llm = _CountingLLM(shared_response)
    shared_judge = LLMJudge(shared_llm, cache_path=str(tmp_path / "shared.sqlite3"))
    other_judge = LLMJudge(
        _MockLLM('{"LCS": 0.6, "ESS": 0.6, "GRS": 0.4, "COMP": 0.6, "BIAS": 0.4, "confidence": 0.7}'),
        cache_path=str(tmp_path / "other.sqlite3"),
    )
    ensemble = EnsembleJudge([shared_judge, other_judge])

    # confidence_threshold above any possible confidence -> always escalates
    escalating = EscalatingJudge(shared_judge, ensemble, confidence_threshold=1.1)
    escalating.compute_reward(CLAIM, REASONING, CLEAN_EVIDENCE)

    # tier-1's own get_scores() call, then the ensemble's matching member
    # hits the cache instead of calling the model again
    assert shared_llm.calls == 1
