"""
Smoke tests for EvidenceLabeler: mirrors test_llm_judge_smoke.py's approach
of a mock LLM client, no real model loading.
"""
import logging

from evid_rl_env.judge import cache as cache_module
from evid_rl_env.judge.evidence_labeler import EvidenceLabeler


def _labeler(mock_llm, response, **kwargs):
    return EvidenceLabeler(mock_llm(response, **kwargs), cache_scores=False)


def test_label_support(mock_llm):
    labeler = _labeler(mock_llm, '{"label": "support", "adversarial": false}')
    assert labeler.label("claim", "text") == "support"


def test_label_contradict(mock_llm):
    labeler = _labeler(mock_llm, '{"label": "contradict", "adversarial": false}')
    assert labeler.label("claim", "text") == "contradict"


def test_label_neutral(mock_llm):
    labeler = _labeler(mock_llm, '{"label": "neutral", "adversarial": false}')
    assert labeler.label("claim", "text") == "neutral"


def test_adversarial_overrides_stance(mock_llm):
    labeler = _labeler(mock_llm, '{"label": "support", "adversarial": true}')
    assert labeler.label("claim", "text") == "adversarial"


def test_unknown_label_value_falls_back_to_neutral(mock_llm):
    labeler = _labeler(mock_llm, '{"label": "something_else", "adversarial": false}')
    assert labeler.label("claim", "text") == "neutral"


def test_malformed_response_falls_back_to_neutral(mock_llm, caplog):
    labeler = _labeler(mock_llm, "not json at all")
    with caplog.at_level(logging.WARNING):
        result = labeler.label("claim", "text")
    assert result == "neutral"
    assert any("could not find a JSON object" in r.message for r in caplog.records)


def test_llm_exception_falls_back_to_neutral(mock_llm, caplog):
    labeler = EvidenceLabeler(mock_llm(raise_exc=RuntimeError("boom")), cache_scores=False)
    with caplog.at_level(logging.WARNING):
        result = labeler.label("claim", "text")
    assert result == "neutral"
    assert any("generation failed" in r.message for r in caplog.records)


def test_caching_avoids_a_second_llm_call(mock_llm, tmp_path):
    llm = mock_llm('{"label": "support", "adversarial": false}')
    labeler = EvidenceLabeler(llm, cache_scores=True)
    # Redirect to a tmp cache so this test doesn't touch or depend on the real
    # artifacts/cache/evidence_label_cache.sqlite3.
    labeler._cache = cache_module.SQLiteCache(str(tmp_path / "cache.sqlite3"))

    labeler.label("claim", "text")
    labeler.label("claim", "text")

    assert llm._call_count == 1
