"""
Unit tests for GoldEvaluator's aggregation math (agent/gold_evaluator.py),
against a scripted fake env/policy/judge — no real model loads, mirroring
the mocked-LLM style used in tests/test_llm_judge_smoke.py.
"""

from types import SimpleNamespace

import pytest

from evid_rl_env.agent.gold_evaluator import GoldEvaluator
from evid_rl_env.environment.state import Evidence

DIMENSIONS = ("LCS", "ESS", "GRS", "COMP", "BIAS")


def _state(claim="claim", reasoning="reasoning text", confidence=0.7):
    return SimpleNamespace(
        claim=claim,
        debate_history=[reasoning],
        selected_evidence=[Evidence(id=0, text="ev", label="support")],
        confidence=confidence,
    )


class _FakeEnv:
    """Each reset()/step() pair plays back one scripted episode."""

    def __init__(self, episodes):
        self.episodes = episodes
        self._i = -1
        self.current_sample = None

    def reset(self):
        self._i += 1
        ep = self.episodes[self._i]
        self.current_sample = {"label": ep["true_label"]}
        self._pending_info = ep
        return _state(confidence=ep["confidence"])

    def step(self, action, payload):
        ep = self._pending_info
        info = {
            "task_success": ep["task_success"],
            "llm_scores": ep["proxy_scores"],
            "llm_reward": ep["proxy_reward"],
        }
        return _state(confidence=ep["confidence"]), 0.0, True, info


class _FakePolicy:
    def act(self, state, greedy=True):
        return "finalize", None, 0


class _FakeGoldJudge:
    def __init__(self, reward, scores):
        self._reward = reward
        self._scores = scores

    def compute_reward(self, claim, reasoning, evidence):
        return self._reward, self._scores


def _scores(v):
    return {k: v for k in DIMENSIONS}


def test_perfect_agreement_zero_disagreement():
    episodes = [
        {"true_label": 1.0, "confidence": 0.9, "task_success": 1.0,
         "proxy_scores": _scores(0.8), "proxy_reward": 0.6},
        {"true_label": 0.0, "confidence": 0.1, "task_success": 1.0,
         "proxy_scores": _scores(0.3), "proxy_reward": 0.3},
    ]
    env = _FakeEnv(episodes)

    # gold judge mirrors whatever proxy scored, so disagreement should be 0
    class _MirrorJudge:
        def compute_reward(self, claim, reasoning, evidence):
            ep = env.episodes[env._i]
            return ep["proxy_reward"], ep["proxy_scores"]

    evaluator = GoldEvaluator(env, _FakePolicy(), [_MirrorJudge()], n_episodes=2)
    result = evaluator.evaluate()

    assert result["n_scored"] == 2
    for dim in DIMENSIONS:
        assert result["dimensions"][dim]["mean_abs_disagreement"] == pytest.approx(0.0)
    assert result["proxy_reward_mean"] == pytest.approx(0.45)
    assert result["gold_reward_mean"] == pytest.approx(0.45)


def test_disagreement_equals_fixed_offset():
    episodes = [
        {"true_label": 1.0, "confidence": 0.9, "task_success": 1.0,
         "proxy_scores": _scores(0.8), "proxy_reward": 0.6},
        {"true_label": 0.0, "confidence": 0.2, "task_success": 1.0,
         "proxy_scores": _scores(0.8), "proxy_reward": 0.6},
    ]
    env = _FakeEnv(episodes)
    # gold judge is consistently 0.3 lower than proxy on every dimension
    gold_judge = _FakeGoldJudge(reward=0.3, scores=_scores(0.5))

    evaluator = GoldEvaluator(env, _FakePolicy(), [gold_judge], n_episodes=2)
    result = evaluator.evaluate()

    for dim in DIMENSIONS:
        assert result["dimensions"][dim]["proxy_mean"] == pytest.approx(0.8)
        assert result["dimensions"][dim]["gold_mean"] == pytest.approx(0.5)
        assert result["dimensions"][dim]["mean_abs_disagreement"] == pytest.approx(0.3)


def test_outcome_accuracy():
    episodes = [
        # confidence matches true_label exactly -> accuracy 1.0, hard-correct
        {"true_label": 1.0, "confidence": 1.0, "task_success": 1.0,
         "proxy_scores": _scores(0.5), "proxy_reward": 0.5},
        # confidence is maximally wrong -> accuracy 0.0, hard-incorrect
        {"true_label": 1.0, "confidence": 0.0, "task_success": 1.0,
         "proxy_scores": _scores(0.5), "proxy_reward": 0.5},
    ]
    env = _FakeEnv(episodes)
    gold_judge = _FakeGoldJudge(reward=0.5, scores=_scores(0.5))

    evaluator = GoldEvaluator(env, _FakePolicy(), [gold_judge], n_episodes=2)
    result = evaluator.evaluate()

    assert result["outcome_accuracy"] == pytest.approx(0.5)
    assert result["outcome_accuracy_hard"] == pytest.approx(0.5)


def test_episodes_without_finalize_are_excluded():
    episodes = [
        {"true_label": 1.0, "confidence": 0.9, "task_success": None,
         "proxy_scores": {}, "proxy_reward": 0.0},
        {"true_label": 1.0, "confidence": 0.9, "task_success": 1.0,
         "proxy_scores": _scores(0.7), "proxy_reward": 0.6},
    ]
    env = _FakeEnv(episodes)
    gold_judge = _FakeGoldJudge(reward=0.6, scores=_scores(0.7))

    evaluator = GoldEvaluator(env, _FakePolicy(), [gold_judge], n_episodes=2)
    result = evaluator.evaluate()

    assert result["n_episodes"] == 2
    assert result["n_scored"] == 1


def test_correlation_none_when_constant():
    episodes = [
        {"true_label": 1.0, "confidence": 0.9, "task_success": 1.0,
         "proxy_scores": _scores(0.5), "proxy_reward": 0.5},
        {"true_label": 1.0, "confidence": 0.9, "task_success": 1.0,
         "proxy_scores": _scores(0.5), "proxy_reward": 0.5},
    ]
    env = _FakeEnv(episodes)
    gold_judge = _FakeGoldJudge(reward=0.5, scores=_scores(0.5))

    evaluator = GoldEvaluator(env, _FakePolicy(), [gold_judge], n_episodes=2)
    result = evaluator.evaluate()

    assert result["proxy_gold_correlation"] is None


def test_escalation_rate_none_when_key_absent():
    """Architectures without an escalation tier (LLMJudge, EnsembleJudge)
    never set "escalated" in their scores dict — None, not 0.0, means
    "not applicable" for this architecture."""
    episodes = [
        {"true_label": 1.0, "confidence": 0.9, "task_success": 1.0,
         "proxy_scores": _scores(0.8), "proxy_reward": 0.6},
        {"true_label": 1.0, "confidence": 0.9, "task_success": 1.0,
         "proxy_scores": _scores(0.5), "proxy_reward": 0.5},
    ]
    env = _FakeEnv(episodes)
    gold_judge = _FakeGoldJudge(reward=0.5, scores=_scores(0.5))

    evaluator = GoldEvaluator(env, _FakePolicy(), [gold_judge], n_episodes=2)
    result = evaluator.evaluate()

    assert result["escalation_rate"] is None


def test_escalation_rate_computed_when_key_present():
    def _scores_with_escalation(v, escalated):
        return {**_scores(v), "escalated": escalated}

    episodes = [
        {"true_label": 1.0, "confidence": 0.9, "task_success": 1.0,
         "proxy_scores": _scores_with_escalation(0.8, True), "proxy_reward": 0.6},
        {"true_label": 1.0, "confidence": 0.9, "task_success": 1.0,
         "proxy_scores": _scores_with_escalation(0.5, False), "proxy_reward": 0.5},
        {"true_label": 1.0, "confidence": 0.9, "task_success": 1.0,
         "proxy_scores": _scores_with_escalation(0.5, False), "proxy_reward": 0.5},
        {"true_label": 1.0, "confidence": 0.9, "task_success": 1.0,
         "proxy_scores": _scores_with_escalation(0.5, False), "proxy_reward": 0.5},
    ]
    env = _FakeEnv(episodes)
    gold_judge = _FakeGoldJudge(reward=0.5, scores=_scores(0.5))

    evaluator = GoldEvaluator(env, _FakePolicy(), [gold_judge], n_episodes=4)
    result = evaluator.evaluate()

    assert result["escalation_rate"] == pytest.approx(0.25)


def test_multiple_gold_judges_are_averaged():
    episodes = [
        {"true_label": 1.0, "confidence": 0.9, "task_success": 1.0,
         "proxy_scores": _scores(0.8), "proxy_reward": 0.6},
    ]
    env = _FakeEnv(episodes)
    judge_low = _FakeGoldJudge(reward=0.2, scores=_scores(0.2))
    judge_high = _FakeGoldJudge(reward=0.6, scores=_scores(0.6))

    evaluator = GoldEvaluator(env, _FakePolicy(), [judge_low, judge_high], n_episodes=1)
    result = evaluator.evaluate()

    assert result["gold_reward_mean"] == pytest.approx(0.4)
    for dim in DIMENSIONS:
        assert result["dimensions"][dim]["gold_mean"] == pytest.approx(0.4)
