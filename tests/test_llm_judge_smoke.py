"""
Smoke test for LLMJudge: verifies parse logic and reward computation
without loading the real Qwen model.

Covers:
- Valid non-zero JSON is parsed correctly and produces non-zero scores.
- An all-zero response (template echo) is rejected and falls back to 0.5 neutral.
- compute_reward returns non-zero values when the mock LLM returns real scores.
"""

import pytest

from evid_rl_env.environment.state import Evidence
from evid_rl_env.judge.llm_judge import LLMJudge


class _MockLLM:
    """Minimal stub that returns a fixed JSON string, no model loading."""

    def __init__(self, response: str):
        self._response = response

    def generate(self, prompt):
        return self._response, len(self._response.split())

    def generate_structured(self, prompt, temperature=None):
        return self.generate(prompt)


CLAIM = "Vaccines cause autism."
REASONING = (
    "The claim is false. Multiple large-scale studies found no link between "
    "the MMR vaccine and autism. The original 1998 paper by Wakefield was "
    "retracted due to data fraud."
)
EVIDENCE = [
    Evidence(id=0, text="A 2019 Danish cohort study of 657,461 children found no increased risk of autism after MMR vaccination.", label="contradict"),
    Evidence(id=1, text="The Lancet retracted Wakefield's 1998 paper after the General Medical Council found serious ethical violations and data manipulation.", label="contradict"),
]


def _make_judge(response: str) -> LLMJudge:
    return LLMJudge(_MockLLM(response), cache_scores=False)


def test_parse_valid_non_zero():
    scores_json = '{"LCS": 0.85, "ESS": 0.90, "GRS": 0.10, "COMP": 0.80, "BIAS": 0.05, "confidence": 0.92}'
    judge = _make_judge(scores_json)
    scores = judge.parse(scores_json)
    assert scores["LCS"] == pytest.approx(0.85)
    assert scores["ESS"] == pytest.approx(0.90)
    assert scores["GRS"] == pytest.approx(0.10)
    assert scores["COMP"] == pytest.approx(0.80)
    assert scores["BIAS"] == pytest.approx(0.05)
    assert scores["confidence"] == pytest.approx(0.92)


def test_parse_rejects_all_zero_template_echo():
    template_echo = '{"LCS": 0.0, "ESS": 0.0, "GRS": 0.0, "COMP": 0.0, "BIAS": 0.0, "confidence": 0.0}'
    judge = _make_judge(template_echo)
    scores = judge.parse(template_echo)
    # All-zero template echo must fall back to neutral 0.5 values
    assert scores["LCS"] == pytest.approx(0.5), "Template echo must be rejected"
    assert scores["ESS"] == pytest.approx(0.5), "Template echo must be rejected"


def test_compute_reward_non_zero():
    good_response = '{"LCS": 0.85, "ESS": 0.90, "GRS": 0.10, "COMP": 0.80, "BIAS": 0.05, "confidence": 0.92}'
    judge = _make_judge(good_response)
    reward, scores = judge.compute_reward(CLAIM, REASONING, EVIDENCE)
    assert reward > 0.0, f"Expected non-zero reward, got {reward}"
    assert scores["LCS"] > 0.0
    assert scores["ESS"] > 0.0
    assert scores["COMP"] > 0.0


def test_compute_reward_empty_reasoning_returns_zeros():
    judge = _make_judge("{}")
    reward, scores = judge.compute_reward(CLAIM, "", EVIDENCE)
    assert reward == 0.0
    assert scores["LCS"] == 0.0
    assert scores["ESS"] == 0.0


def test_get_scores_matches_compute_reward_scores():
    """Regression guard for the get_scores() extraction (added so
    EnsembleJudge can fetch raw per-member scores without going through
    compute_reward's reward formula), must return exactly what
    compute_reward's second return value was before the refactor."""
    good_response = '{"LCS": 0.85, "ESS": 0.90, "GRS": 0.10, "COMP": 0.80, "BIAS": 0.05, "confidence": 0.92}'
    judge = _make_judge(good_response)

    scores_direct = judge.get_scores(CLAIM, REASONING, EVIDENCE)
    _, scores_via_compute_reward = judge.compute_reward(CLAIM, REASONING, EVIDENCE)

    assert scores_direct == scores_via_compute_reward


def test_get_scores_empty_reasoning_matches_sentinel():
    judge = _make_judge("{}")
    scores = judge.get_scores(CLAIM, "", EVIDENCE)
    assert scores == {"LCS": 0.0, "ESS": 0.0, "GRS": 0.5, "COMP": 0.0, "BIAS": 0.5, "confidence": 0.0}


def test_parse_recovers_json_wrapped_in_prose():
    judge = _make_judge("")
    wrapped = (
        'Sure! {"LCS": 0.7, "ESS": 0.6, "GRS": 0.2, "COMP": 0.5, "BIAS": 0.1, '
        '"confidence": 0.8} hope that helps'
    )
    assert judge.parse(wrapped)["LCS"] == pytest.approx(0.7)


def test_parse_unparseable_response_gives_neutral_scores():
    judge = _make_judge("")
    scores = judge.parse("no scores here")
    assert scores == {"LCS": 0.5, "ESS": 0.5, "GRS": 0.5, "COMP": 0.5, "BIAS": 0.5,
                      "confidence": 0.5}


def test_all_neutral_scores_map_to_neutral_reward_not_the_formula_value():
    neutral = {"LCS": 0.5, "ESS": 0.5, "GRS": 0.5, "COMP": 0.5, "BIAS": 0.5, "confidence": 0.9}
    assert LLMJudge._scores_to_reward(neutral) == 0.5


def test_scores_to_reward_pinned_value():
    scores = {"LCS": 0.8, "ESS": 0.8, "GRS": 0.2, "COMP": 0.8, "BIAS": 0.2, "confidence": 1.0}
    assert LLMJudge._scores_to_reward(scores) == pytest.approx(0.30 * 0.8 + 0.25 * 0.8
                                                               + 0.20 * 0.8 - 0.25 * 0.2
                                                               - 0.15 * 0.2)
