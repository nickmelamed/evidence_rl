"""
Verifies LLMJudge.cache_path keeps separate judge instances from colliding.

Before this parameter existed, every LLMJudge shared the single hardcoded
"artifacts/cache/judge_cache.sqlite3" path — since the cache key is a content
hash of (claim, reasoning, evidence) and doesn't include the model name, a
second judge (e.g. a gold-eval judge pointed at a different model) scoring
the same trajectory would silently read back the first judge's cached
scores instead of generating its own.
"""

from evid_rl_env.environment.state import Evidence
from evid_rl_env.judge.llm_judge import LLMJudge

CLAIM = "Water boils at 100C at sea level."
REASONING = "This is well established physics."
EVIDENCE = [Evidence(id=0, text="Standard atmospheric pressure boiling point is 100C.", label="support")]


class _MockLLM:
    def __init__(self, response: str):
        self._response = response

    def generate(self, prompt):
        return self._response, len(self._response.split())

    def generate_structured(self, prompt, temperature=None):
        return self.generate(prompt)


def test_different_cache_paths_do_not_collide(tmp_path):
    response_a = '{"LCS": 0.9, "ESS": 0.9, "GRS": 0.1, "COMP": 0.9, "BIAS": 0.1, "confidence": 0.9}'
    response_b = '{"LCS": 0.2, "ESS": 0.2, "GRS": 0.8, "COMP": 0.2, "BIAS": 0.8, "confidence": 0.9}'

    judge_a = LLMJudge(_MockLLM(response_a), cache_path=str(tmp_path / "a.sqlite3"))
    judge_b = LLMJudge(_MockLLM(response_b), cache_path=str(tmp_path / "b.sqlite3"))

    reward_a, scores_a = judge_a.compute_reward(CLAIM, REASONING, EVIDENCE)
    reward_b, scores_b = judge_b.compute_reward(CLAIM, REASONING, EVIDENCE)

    assert scores_a["LCS"] == 0.9
    assert scores_b["LCS"] == 0.2
    assert reward_a != reward_b


def test_same_cache_path_shares_cache(tmp_path):
    """Sanity check the isolation test above is actually exercising the
    cache and not just two independent judges that never collide anyway."""
    shared_path = str(tmp_path / "shared.sqlite3")
    response_a = '{"LCS": 0.9, "ESS": 0.9, "GRS": 0.1, "COMP": 0.9, "BIAS": 0.1, "confidence": 0.9}'
    response_b = '{"LCS": 0.2, "ESS": 0.2, "GRS": 0.8, "COMP": 0.2, "BIAS": 0.8, "confidence": 0.9}'

    judge_a = LLMJudge(_MockLLM(response_a), cache_path=shared_path)
    judge_b = LLMJudge(_MockLLM(response_b), cache_path=shared_path)

    judge_a.compute_reward(CLAIM, REASONING, EVIDENCE)
    # same content hash, shared cache file -> judge_b reads judge_a's cached
    # scores instead of generating its own from response_b
    _, scores_b = judge_b.compute_reward(CLAIM, REASONING, EVIDENCE)

    assert scores_b["LCS"] == 0.9
