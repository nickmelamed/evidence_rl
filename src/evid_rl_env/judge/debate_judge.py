import logging

from evid_rl_env.judge.ensemble_judge import _sanitize
from evid_rl_env.judge.llm_judge import LLMJudge

_logger = logging.getLogger(__name__)

_DEFAULT_ADVOCATE_FOR_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
_DEFAULT_ADVOCATE_AGAINST_MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
_DEFAULT_ARBITER_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"


def build_debate_judge(
    seed: int,
    advocate_for_model: str = _DEFAULT_ADVOCATE_FOR_MODEL,
    advocate_against_model: str = _DEFAULT_ADVOCATE_AGAINST_MODEL,
    arbiter_model: str = _DEFAULT_ARBITER_MODEL,
    reuse: dict = None,
) -> "DebateJudge":
    """`reuse` (model_name -> already-built LLMJudge) lets a caller hand in
    an instance it already has for a given model name instead of loading a
    fresh one — used by environment.py to share the underlying model client
    between EscalatingJudge's tier-1 and a matching advocate/arbiter role.
    Unlike EnsembleJudge's reuse, this only saves a duplicate model *load*,
    not a duplicate call: the advocate prompts ask for a persuasive
    argument, a genuinely different task from tier-1's JSON-scoring prompt
    even on the same model, so there's no real cache hit to gain here —
    only the arbiter is wrapped in an LLMJudge, and it uses its own
    debate-specific cache path (its input is the augmented reasoning plus
    both critiques, a different cache key space than tier-1's)."""
    from evid_rl_env.agent.llm_client import JudgeLLMClient

    reuse = reuse or {}

    def _client_for(model_name):
        if model_name in reuse:
            return reuse[model_name].llm
        return JudgeLLMClient(model_name=model_name, seed=seed)

    advocate_for = _client_for(advocate_for_model)
    advocate_against = _client_for(advocate_against_model)
    arbiter = LLMJudge(
        _client_for(arbiter_model),
        cache_path=f"artifacts/cache/judge_cache_debate_arbiter_{_sanitize(arbiter_model)}.sqlite3",
    )
    return DebateJudge(advocate_for, advocate_against, arbiter)


def _evidence_text(evidence: list) -> str:
    return "\n".join(f"- {e.text}" for e in evidence)


def _advocate_for_prompt(claim, reasoning, evidence) -> str:
    return (
        "You are an advocate arguing that the REASONING below is HIGH "
        "QUALITY. Given the claim, evidence, and reasoning, write the "
        "strongest honest case for why this reasoning is logically "
        "consistent, well-supported by the evidence, properly grounded, "
        "complete, and unbiased. Cite specifics.\n\n"
        f"CLAIM:\n{claim}\n\nEVIDENCE:\n{_evidence_text(evidence)}\n\n"
        f"REASONING:\n{reasoning}\n\n"
        "Your argument FOR this reasoning being high quality:"
    )


def _advocate_against_prompt(claim, reasoning, evidence) -> str:
    return (
        "You are a critic arguing that the REASONING below is LOW "
        "QUALITY or FLAWED. Given the claim, evidence, and reasoning, "
        "write the strongest honest case for why this reasoning has "
        "logical gaps, is poorly supported by the evidence, introduces "
        "ungrounded claims, is incomplete, or is biased. Cite specifics.\n\n"
        f"CLAIM:\n{claim}\n\nEVIDENCE:\n{_evidence_text(evidence)}\n\n"
        f"REASONING:\n{reasoning}\n\n"
        "Your argument AGAINST this reasoning being high quality:"
    )


class DebateJudge:
    """Two adversarial critics argue for/against the reasoning's quality;
    a separate, stronger arbiter model scores after seeing both critiques.
    The most expensive judge architecture in this stack (3 model calls) —
    intended as the escalation hierarchy's top tier (see
    judge/escalating_judge.py), not the default per-step path. Same
    (claim, reasoning, evidence) -> (reward, scores) interface as
    LLMJudge/EnsembleJudge.

    The arbiter reuses LLMJudge's existing scoring prompt/parse/cache/
    reward-formula machinery unchanged: the two critiques are appended to
    `reasoning` as extra labeled context before being handed to a plain
    LLMJudge.compute_reward call, rather than building a new prompt format
    for the arbiter from scratch.
    """

    def __init__(self, advocate_for, advocate_against, arbiter: LLMJudge):
        self.advocate_for = advocate_for
        self.advocate_against = advocate_against
        self.arbiter = arbiter

    def _critique(self, llm, role: str, prompt: str) -> str:
        try:
            if hasattr(llm, "generate_structured"):
                text, _ = llm.generate_structured(prompt)
            else:
                text, _ = llm.generate(prompt)
            return text
        except Exception as exc:
            _logger.warning(
                "DebateJudge: %s critique generation failed (%s) | model=%s — "
                "using empty critique.",
                role, exc, getattr(llm, "model_name", "unknown"),
            )
            return ""

    def compute_reward(self, claim, reasoning, evidence):
        if not reasoning.strip():
            return 0.0, {"LCS": 0.0, "ESS": 0.0, "GRS": 0.5, "COMP": 0.0, "BIAS": 0.5, "confidence": 0.0}

        critique_for = self._critique(
            self.advocate_for, "advocate-for", _advocate_for_prompt(claim, reasoning, evidence)
        )
        critique_against = self._critique(
            self.advocate_against, "advocate-against", _advocate_against_prompt(claim, reasoning, evidence)
        )

        augmented_reasoning = (
            f"{reasoning}\n\n"
            f"--- Argument FOR this reasoning being high quality ---\n{critique_for}\n\n"
            f"--- Argument AGAINST this reasoning being high quality ---\n{critique_against}"
        )
        return self.arbiter.compute_reward(claim, augmented_reasoning, evidence)
