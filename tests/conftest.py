"""Shared fixtures for the test suite.

Design goal: exercise environment/reward/judge/baseline logic without ever
loading a real generation model (gemma/qwen via transformers.pipeline). This
is possible because:
  - judge/llm_judge.py has no ML dependency (stdlib + numpy + sqlite3), so a
    real LLMJudge can be built around a lightweight _MockLLM and injected
    straight into ClaimEnv via its `llm_judge=` constructor parameter.
  - agent/baseline.py's baseline classes take a duck-typed `llm_client`
    directly, so _MockLLM can stand in for GreedyLLMBaseline/FewShotLLMBaseline/
    BestOfNBaseline too.
  - Tests never construct a real ActorCriticPolicy/LLMClient, so
    transformers/torch's heavy generation pipelines are never touched. Tests
    that exercise PPO/PolicyGradient/the trainers do still import
    agent.policy for `encode_state`, which (via state_encoder.py) loads the
    much lighter sentence-transformers embedding model — a one-time, fully
    offline-after-first-download cost, not the multi-GB generation models.
"""

import numpy as np
import pytest

from evid_rl_env.environment.actions import ACTIONS, Actions
from evid_rl_env.environment.environment import ClaimEnv
from evid_rl_env.environment.state import Evidence
from evid_rl_env.judge.llm_judge import LLMJudge

GOOD_SCORES_JSON = (
    '{"LCS": 0.8, "ESS": 0.8, "GRS": 0.2, "COMP": 0.8, "BIAS": 0.2, "confidence": 0.9}'
)


class _MockLLM:
    """Minimal stub LLM client: no model loading, fully deterministic.

    Pass `responses` (a list) to return a different canned response on each
    successive call (last one repeats once the list is exhausted); pass
    `raise_exc` to make every call raise, for testing fallback paths.
    """

    def __init__(self, response: str = GOOD_SCORES_JSON, responses=None,
                 model_name: str = "mock-llm", raise_exc: Exception = None):
        self._response = response
        self._responses = list(responses) if responses is not None else None
        self._call_count = 0
        self.model_name = model_name
        self._raise_exc = raise_exc

    def generate(self, prompt):
        self._call_count += 1
        if self._raise_exc is not None:
            raise self._raise_exc
        if self._responses is not None:
            idx = min(self._call_count - 1, len(self._responses) - 1)
            resp = self._responses[idx]
        else:
            resp = self._response
        return resp, len(resp.split())

    def generate_structured(self, prompt, temperature=None):
        return self.generate(prompt)


@pytest.fixture
def mock_llm():
    return _MockLLM


@pytest.fixture
def fake_evidence_docs():
    """Raw Tavily-shaped dicts, as returned by evidence_fetcher.fetch_evidence."""
    return [
        {"content": "A large randomized trial found a supporting effect.", "score": 0.9},
        {"content": "A cohort study found a contradicting result.", "score": 0.8},
        {"content": "An unrelated background article.", "score": 0.5},
    ]


@pytest.fixture(autouse=True)
def no_network_fetch(monkeypatch, fake_evidence_docs):
    """Prevent every test from making a real Tavily network call.

    fetch_evidence is looked up from two places at runtime: environment.py's
    module-level import (used by reset()) and a fresh import inside the QUERY
    action handler (used mid-episode) — both must be patched.
    """
    def _fake_fetch(claim, search_query, max_results=5):
        return list(fake_evidence_docs)

    monkeypatch.setattr("evid_rl_env.environment.environment.fetch_evidence", _fake_fetch)
    monkeypatch.setattr("evid_rl_env.data.evidence_fetcher.fetch_evidence", _fake_fetch)


@pytest.fixture
def evidence_pool():
    return [
        Evidence(id=0, text="Supporting evidence for the claim.", label="support"),
        Evidence(id=1, text="Contradicting evidence for the claim.", label="contradict"),
        Evidence(id=2, text="Neutral background evidence.", label="neutral"),
    ]


@pytest.fixture
def make_llm_judge():
    def _make(response: str = GOOD_SCORES_JSON, **kwargs) -> LLMJudge:
        return LLMJudge(_MockLLM(response, **kwargs), cache_scores=False)
    return _make


class FakeLabeler:
    """Injectable stand-in for EvidenceLabeler with fully controlled, static
    labels — avoids ever constructing a real EvidenceLabeler/JudgeLLMClient.
    Defaults every text to "neutral", matching the pre-labeling-pipeline
    behavior every text used to get, so existing tests that don't care about
    labels are unaffected."""

    def __init__(self, labels_by_text: dict = None, default: str = "neutral"):
        self._labels_by_text = labels_by_text or {}
        self._default = default

    def label(self, claim: str, text: str) -> str:
        return self._labels_by_text.get(text, self._default)


@pytest.fixture
def fake_evidence_labeler():
    return FakeLabeler


@pytest.fixture
def make_env(make_llm_judge, fake_evidence_labeler):
    """Build a ClaimEnv with an injected mock judge — no torch, no network.

    Pass `embedder=<callable>` to also inject a fake embedder (see
    ClaimEnv's `embedder=` param) instead of the real sentence-transformers
    one, keeping RERANK tests torch-free too. Pass `evidence_labeler=<obj>`
    (e.g. a FakeLabeler instance) to control per-text stance/reliability
    labels instead of the all-"neutral" default.
    """
    def _make(mock_response: str = GOOD_SCORES_JSON, dataset=None, seed: int = 0,
              embedder=None, evidence_labeler=None, **mock_kwargs):
        dataset = dataset if dataset is not None else [
            {"claim": "Test claim for the episode.", "search_query": "test claim"}
        ]
        judge = make_llm_judge(mock_response, **mock_kwargs)
        labeler = evidence_labeler if evidence_labeler is not None else fake_evidence_labeler()
        return ClaimEnv(dataset, llm_judge=judge, seed=seed, embedder=embedder,
                         evidence_labeler=labeler)
    return _make


class FakePolicy:
    """Duck-typed stand-in for ActorCriticPolicy with plain numpy params —
    avoids ever constructing a real LLMClient (no gemma/qwen model load)."""

    def __init__(self, state_dim: int = 4, n_actions: int = None, seed: int = 0):
        n_actions = n_actions if n_actions is not None else len(ACTIONS)
        rng = np.random.RandomState(seed)
        self.actions = list(ACTIONS)
        self.n_actions = n_actions
        self.state_dim = state_dim
        self.actor_params = rng.randn(state_dim, n_actions) * 0.01
        self.value_params = np.zeros(state_dim)
        self.last_entropy = 0.0
        self.last_probs = np.ones(n_actions) / n_actions
        self.llm = None  # no .llm -> Trainer/BanditTrainer skip baseline construction

    def reset_episode_cache(self):
        pass

    def _features(self, state):
        return np.array([
            len(state.selected_evidence),
            len(state.evidence_pool),
            len(state.debate_history),
            state.steps_taken,
        ], dtype=np.float64)

    def get_probs(self, state):
        features = self._features(state)
        logits = features @ self.actor_params
        logits = logits - np.max(logits)
        exp = np.exp(logits)
        return exp / np.sum(exp)

    def get_value(self, state):
        return float(self._features(state) @ self.value_params)

    def act(self, state, greedy: bool = False, force_action_idx: int = None):
        probs = self.get_probs(state)
        self.last_probs = probs.copy()
        self.last_entropy = float(-np.sum(probs * np.log(probs + 1e-8)))

        if force_action_idx is not None:
            idx = force_action_idx
        else:
            idx = int(np.argmax(probs)) if greedy else np.random.choice(self.n_actions, p=probs)
        action = self.actions[idx]

        if action == Actions.SELECT:
            doc = np.random.choice(state.evidence_pool) if state.evidence_pool else None
            return action, (doc.id if doc is not None else 0), idx
        if action == Actions.REMOVE:
            doc = np.random.choice(state.selected_evidence) if state.selected_evidence else None
            return action, (doc.id if doc is not None else 0), idx
        if action in (Actions.SUPPORT, Actions.CONTRADICT):
            return action, {"argument": "test argument", "evidence_ids": [], "tokens": 3}, idx
        if action == Actions.QUERY:
            return action, state.claim, idx
        if action == Actions.SUMMARIZE:
            return action, {"summary": "test summary", "tokens": 2}, idx
        if action == Actions.CONCEDE:
            return action, {"argument": "test concession", "tokens": 2}, idx
        if action == Actions.ASSIGN_CONFIDENCE:
            return action, 0.5, idx
        if action == Actions.CHALLENGE_EVIDENCE:
            target = np.random.choice(state.evidence_pool) if state.evidence_pool else None
            return action, {"evidence_id": target.id if target is not None else 0,
                             "argument": "test challenge", "tokens": 2}, idx
        if action == Actions.REQUEST_CLARIFICATION:
            return action, state.claim, idx
        if action == Actions.HEDGE:
            return action, {"argument": "test hedge", "tokens": 2}, idx
        return action, None, idx

    def grad_log_prob(self, state, action_idx, features=None, probs=None):
        if features is None:
            features = self._features(state)
        if probs is None:
            probs = self.get_probs(state)
        grad = -probs.copy()
        grad[action_idx] += 1
        return np.outer(features, grad)

    def save(self, path: str) -> None:
        np.savez_compressed(path, actor_params=self.actor_params, value_params=self.value_params)


@pytest.fixture
def fake_policy():
    return FakePolicy
