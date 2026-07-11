import hashlib
import json
import logging
import re

from evid_rl_env.judge.cache import SQLiteCache

_logger = logging.getLogger(__name__)

_VALID_STANCES = ("support", "contradict", "neutral")


class EvidenceLabeler:
    """
    Labels a single piece of retrieved evidence's stance toward a claim
    (support/contradict/neutral) and whether it looks adversarial (unreliable
    or low-quality), via a cached LLM call.

    Structurally a sibling of judge.llm_judge.LLMJudge: same llm/cache_scores
    shape, same cache-by-content-hash pattern, same safe-fallback-on-failure
    behavior — labeling failures fall back to "neutral", which is exactly
    today's permanent default, so an outage never makes things worse than the
    pre-labeling baseline.

    `adversarial: true` overrides the stance label in the returned string,
    since judge.metrics's compute_precision/recall/contradiction_acknowledgment/
    adversarial_contamination all check a single categorical Evidence.label
    field for exact string equality — a document can't be both "support" and
    "adversarial" under that existing single-field taxonomy.
    """

    def __init__(self, llm, cache_scores: bool = True):
        self.llm = llm
        self.cache_scores = cache_scores
        self._cache = SQLiteCache("artifacts/cache/evidence_label_cache.sqlite3")

    def _cache_key(self, claim: str, text: str) -> str:
        return hashlib.md5((claim + text).encode()).hexdigest()

    def build_prompt(self, claim: str, text: str) -> str:
        return (
            "You are labeling one piece of retrieved evidence for a claim-"
            "verification task. Respond with ONLY a JSON object — no "
            "explanation, no markdown, no preamble.\n\n"
            f"CLAIM:\n{claim}\n\n"
            f"EVIDENCE:\n{text}\n\n"
            'Respond with exactly this JSON shape: {"label": ..., "adversarial": ...}\n\n'
            "label = one of \"support\" (evidence supports the claim), "
            "\"contradict\" (evidence contradicts the claim), or "
            "\"neutral\" (related but neither clearly supports nor contradicts)\n"
            "adversarial = true if the evidence appears unreliable, low-quality, "
            "or from a source likely to be biased or misleading; false otherwise\n\n"
            "Respond with the JSON object only, with your real answer filled in:"
        )

    def _parse(self, response: str) -> str:
        try:
            data = json.loads(response.strip())
        except (json.JSONDecodeError, AttributeError, ValueError):
            match = re.search(r'\{[^{}]*"label"[^{}]*\}', response, re.DOTALL)
            if not match:
                _logger.warning(
                    "EvidenceLabeler.parse: could not find a JSON object in "
                    "response %r — using neutral fallback.", response[:120],
                )
                return "neutral"
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                _logger.warning(
                    "EvidenceLabeler.parse: malformed JSON %r — using neutral fallback.",
                    match.group()[:120],
                )
                return "neutral"

        if bool(data.get("adversarial", False)) is True:
            return "adversarial"

        label = data.get("label", "neutral")
        return label if label in _VALID_STANCES else "neutral"

    def label(self, claim: str, text: str) -> str:
        """Returns one of: support, contradict, neutral, adversarial."""
        if self.cache_scores:
            key = self._cache_key(claim, text)
            if key in self._cache:
                return self._cache[key]

        prompt = self.build_prompt(claim, text)
        try:
            if hasattr(self.llm, "generate_structured"):
                response, _ = self.llm.generate_structured(prompt)
            else:
                response, _ = self.llm.generate(prompt)
            result = self._parse(response)
        except Exception as exc:
            _logger.warning(
                "EvidenceLabeler: generation failed (%s) | model=%s | claim='%.80s' — "
                "using neutral fallback.",
                exc, getattr(self.llm, "model_name", "unknown"), claim,
            )
            result = "neutral"

        if self.cache_scores:
            self._cache[key] = result
        return result
