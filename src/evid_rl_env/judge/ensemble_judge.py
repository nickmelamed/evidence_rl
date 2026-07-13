import numpy as np

from evid_rl_env.judge.llm_judge import LLMJudge

_DIMENSIONS = ("LCS", "ESS", "GRS", "COMP", "BIAS")
# Maximum possible std for two values bounded in [0,1] is 0.5 (one member at
# 0, the other at 1) — this normalization is calibrated for exactly 2
# members and would need revisiting if the ensemble grows past that.
_MAX_PAIRWISE_STD = 0.5


def _sanitize(model_name: str) -> str:
    return model_name.replace("/", "__")


def build_ensemble_judge(model_names: list, seed: int, reuse: dict = None) -> "EnsembleJudge":
    """One JudgeLLMClient + LLMJudge per model name, each with its own
    sanitized cache path so members never collide (see LLMJudge.cache_path,
    added for exactly this reason in the gold-eval harness).

    `reuse` (model_name -> already-built LLMJudge) lets a caller hand in an
    instance it already has for a given model name instead of building a
    fresh one — used by environment.py to share the same LLMJudge between
    EscalatingJudge's tier-1 and a matching ensemble member, which also
    means a genuine cache hit (not just a saved model load) on escalation:
    tier-1 already scored this exact (claim, reasoning, evidence) and wrote
    it to the shared instance's cache moments earlier, so the "duplicate"
    member call returns instantly instead of re-querying the model."""
    from evid_rl_env.agent.llm_client import JudgeLLMClient

    reuse = reuse or {}
    judges = []
    for name in model_names:
        if name in reuse:
            judges.append(reuse[name])
            continue
        client = JudgeLLMClient(model_name=name, seed=seed)
        judges.append(LLMJudge(client, cache_path=f"artifacts/cache/judge_cache_{_sanitize(name)}.sqlite3"))
    return EnsembleJudge(judges)


class EnsembleJudge:
    """Aggregates multiple LLMJudge members' scores (median per dimension)
    and replaces each member's self-reported confidence with *measured*
    inter-judge agreement — a policy can't inflate reward by learning
    reasoning that makes a single judge simply claim high confidence,
    since this confidence is derived from actual cross-model agreement
    instead. Same (claim, reasoning, evidence) -> (reward, scores)
    interface as LLMJudge, so it's a drop-in wherever
    self.llm_judge.compute_reward is called (see environment.py)."""

    def __init__(self, judges: list):
        if len(judges) < 2:
            raise ValueError("EnsembleJudge requires at least 2 member judges")
        self.judges = judges

    def compute_reward(self, claim, reasoning, evidence):
        if not reasoning.strip():
            return 0.0, {"LCS": 0.0, "ESS": 0.0, "GRS": 0.5, "COMP": 0.0, "BIAS": 0.5, "confidence": 0.0}

        member_scores = [j.get_scores(claim, reasoning, evidence) for j in self.judges]

        aggregated = {}
        stds = []
        for dim in _DIMENSIONS:
            values = [float(s.get(dim, 0.5)) for s in member_scores]
            aggregated[dim] = float(np.median(values))
            stds.append(float(np.std(values)))

        mean_disagreement = float(np.mean(stds))
        aggregated["confidence"] = float(np.clip(1.0 - mean_disagreement / _MAX_PAIRWISE_STD, 0.0, 1.0))

        reward = LLMJudge._scores_to_reward(aggregated)
        return reward, aggregated
