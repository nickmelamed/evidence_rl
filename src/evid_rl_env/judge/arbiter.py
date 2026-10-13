from evid_rl_env.judge.llm_judge import LLMJudge


class Arbiter:
    """Score a two-sided debate transcript and issue the reward both roles share.

    It wraps an ``LLMJudge`` so it can stand in wherever the environment calls
    ``compute_reward``. The transcript already carries the ``SUPPORT:`` and
    ``CONTRADICT:`` prefixes that ``ClaimEnv`` writes, so the judge sees which
    side said what.
    """

    def __init__(self, judge: LLMJudge, model_name: str):
        self.judge = judge
        self.model_name = model_name

    def compute_reward(self, claim, reasoning, evidence):
        return self.judge.compute_reward(claim, reasoning, evidence)


def build_arbiter(model_name: str, seed: int) -> Arbiter:
    """Build an arbiter around ``model_name`` with its own score cache.

    The cache key hashes the claim, reasoning and evidence but not the model, so
    sharing the default cache file would return another judge's scores.
    """
    if not model_name:
        raise ValueError("arbiter_model must name a model")
    from evid_rl_env.agent.llm_client import JudgeLLMClient

    client = JudgeLLMClient(model_name=model_name, seed=seed)
    cache_path = f"artifacts/cache/judge_cache_{model_name.replace('/', '__')}.sqlite3"
    return Arbiter(LLMJudge(client, cache_path=cache_path), model_name=model_name)
