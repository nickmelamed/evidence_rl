"""
Unit tests for DebateJudge (judge/debate_judge.py): augmented-reasoning
composition, empty-reasoning short-circuit, graceful degradation on a
failed critique, and pass-through of the arbiter's (reward, scores) —
all against mocked clients, no real model loads.
"""

import pytest

from evid_rl_env.environment.state import Evidence
from evid_rl_env.judge.debate_judge import DebateJudge
from evid_rl_env.judge.llm_judge import LLMJudge

CLAIM = "The Earth orbits the Sun."
REASONING = "Well-established heliocentric astronomy confirms this."
EVIDENCE = [Evidence(id=0, text="Kepler's laws describe planetary orbits around the Sun.", label="support")]


class _StubCritic:
    def __init__(self, response: str = "critique text"):
        self._response = response

    def generate(self, prompt):
        return self._response, len(self._response.split())

    def generate_structured(self, prompt, temperature=None):
        return self.generate(prompt)


class _RaisingCritic:
    def generate(self, prompt):
        raise AssertionError("should not be called")

    def generate_structured(self, prompt, temperature=None):
        raise AssertionError("should not be called")


class _FailingCritic:
    def generate(self, prompt):
        raise RuntimeError("model exploded")

    def generate_structured(self, prompt, temperature=None):
        raise RuntimeError("model exploded")


class _CapturingArbiter:
    """Captures the reasoning it was asked to score, returns a fixed result."""

    def __init__(self, reward=0.6, scores=None):
        self._reward = reward
        self._scores = scores or {"LCS": 0.6, "ESS": 0.6, "GRS": 0.4, "COMP": 0.6, "BIAS": 0.4, "confidence": 0.7}
        self.received_reasoning = None
        self.call_count = 0

    def compute_reward(self, claim, reasoning, evidence):
        self.received_reasoning = reasoning
        self.call_count += 1
        return self._reward, dict(self._scores)


class _RaisingArbiter:
    def compute_reward(self, claim, reasoning, evidence):
        raise AssertionError("should not be called")


def test_augmented_reasoning_includes_both_critiques_and_original(tmp_path):
    for_critic = _StubCritic("Argument FOR: well grounded and complete.")
    against_critic = _StubCritic("Argument AGAINST: overstates certainty.")
    arbiter = _CapturingArbiter()

    debate = DebateJudge(for_critic, against_critic, arbiter)
    debate.compute_reward(CLAIM, REASONING, EVIDENCE)

    assert arbiter.call_count == 1
    assert REASONING in arbiter.received_reasoning
    assert "Argument FOR: well grounded and complete." in arbiter.received_reasoning
    assert "Argument AGAINST: overstates certainty." in arbiter.received_reasoning


def test_empty_reasoning_short_circuits_without_calling_any_component():
    debate = DebateJudge(_RaisingCritic(), _RaisingCritic(), _RaisingArbiter())

    reward, scores = debate.compute_reward(CLAIM, "", EVIDENCE)

    assert reward == 0.0
    assert scores["LCS"] == 0.0


def test_failed_critique_degrades_to_empty_string_and_still_proceeds():
    against_critic = _StubCritic("Argument AGAINST: has gaps.")
    arbiter = _CapturingArbiter()

    debate = DebateJudge(_FailingCritic(), against_critic, arbiter)
    debate.compute_reward(CLAIM, REASONING, EVIDENCE)

    assert arbiter.call_count == 1
    assert "Argument AGAINST: has gaps." in arbiter.received_reasoning
    # the failed FOR critique contributes an empty section, not a crash
    assert "Argument FOR this reasoning being high quality ---\n\n" in arbiter.received_reasoning


def test_final_reward_and_scores_are_the_arbiters_pass_through():
    arbiter_scores = {"LCS": 0.3, "ESS": 0.3, "GRS": 0.7, "COMP": 0.3, "BIAS": 0.7, "confidence": 0.9}
    arbiter = _CapturingArbiter(reward=0.15, scores=arbiter_scores)
    debate = DebateJudge(_StubCritic(), _StubCritic(), arbiter)

    reward, scores = debate.compute_reward(CLAIM, REASONING, EVIDENCE)

    assert reward == 0.15
    assert scores == arbiter_scores


def test_build_debate_judge_uses_reuse_client(monkeypatch, tmp_path):
    """A role whose model name is present in `reuse` must reuse that
    LLMJudge's underlying client (saving a duplicate model load) instead
    of building a fresh one."""
    from evid_rl_env.judge.debate_judge import build_debate_judge

    built_clients = []

    class _FakeJudgeLLMClient:
        def __init__(self, model_name, seed):
            built_clients.append(model_name)
            self.model_name = model_name

    monkeypatch.setattr("evid_rl_env.agent.llm_client.JudgeLLMClient", _FakeJudgeLLMClient)

    reused_client = _StubCritic()
    reused_judge = LLMJudge(reused_client, cache_path=str(tmp_path / "reused.sqlite3"))

    debate = build_debate_judge(
        seed=1,
        advocate_for_model="model-for",
        advocate_against_model="model-against",
        arbiter_model="model-arbiter",
        reuse={"model-for": reused_judge},
    )

    assert debate.advocate_for is reused_client
    assert built_clients == ["model-against", "model-arbiter"]  # model-for never built fresh
