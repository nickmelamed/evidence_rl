import hashlib
import json
import logging
import re

import numpy as np

from evid_rl_env.judge.cache import SQLiteCache

_logger = logging.getLogger(__name__)


class LLMJudge:
    def __init__(self, llm, cache_scores=True):
        self.llm = llm
        self.cache_scores = cache_scores
        self._cache = SQLiteCache("artifacts/cache/judge_cache.sqlite3")

    def _cache_key(self, claim, reasoning, evidence):
        raw = "\x1f".join([claim, reasoning, *(e.text for e in evidence)])
        return hashlib.md5(raw.encode()).hexdigest()

    def build_prompt(self, claim, reasoning, evidence):
        evidence_text = "\n".join([f"- {e.text}" for e in evidence])
        return (
            "You are an expert evaluator of reasoning quality. "
            "You must respond with ONLY a JSON object — no explanation, "
            "no markdown, no preamble.\n\n"
            f"CLAIM:\n{claim}\n\n"
            f"EVIDENCE:\n{evidence_text}\n\n"
            f"REASONING:\n{reasoning}\n\n"
            "Score the reasoning on exactly these five dimensions. "
            "Replace each ... with your actual numeric score (float between 0 and 1):\n"
            '{"LCS": ..., "ESS": ..., "GRS": ..., "COMP": ..., "BIAS": ..., "confidence": ...}\n\n'
            "LCS = logical consistency (1=fully coherent)\n"
            "ESS = evidence support (1=correctly uses all evidence)\n"
            "GRS = grounding risk (1=reasoning introduces claims not in evidence, 0=fully grounded)\n"
            "COMP = completeness (1=fully addresses the claim)\n"
            "BIAS = selective citation bias (1=only cites supporting evidence while ignoring contradictions it was given, 0=balanced)\n"
            "confidence = your confidence in these scores (0-1)\n\n"
            "Respond with the JSON object only, with your real scores filled in:"
        )

    def _is_template_echo(self, data: dict) -> bool:
        """Return True if the model echoed the prompt template (all scoring keys are 0.0)."""
        scoring_keys = ("LCS", "ESS", "GRS", "COMP", "BIAS")
        return all(float(data.get(k, 1.0)) == 0.0 for k in scoring_keys)

    _NEUTRAL = {"LCS": 0.5, "ESS": 0.5, "GRS": 0.5, "COMP": 0.5, "BIAS": 0.5, "confidence": 0.5}

    def parse(self, response):
        # Direct JSON parse: accept only if not a template echo.
        try:
            data = json.loads(response.strip())
            if not self._is_template_echo(data):
                return data
            _logger.warning("LLMJudge.parse: all-zero template echo detected — using neutral fallback.")
            return dict(self._NEUTRAL)
        except (json.JSONDecodeError, AttributeError, ValueError):
            pass

        # Regex extraction of the first JSON-object containing "LCS".
        match = re.search(r'\{[^{}]*"LCS"[^{}]*\}', response, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                if not self._is_template_echo(data):
                    return data
            except json.JSONDecodeError:
                pass

        # Key-by-key regex fallback.
        scores = {}
        for key in ["LCS", "ESS", "GRS", "COMP", "BIAS", "confidence"]:
            m = re.search(rf'"{key}"\s*:\s*([0-9.]+)', response)
            if m:
                try:
                    scores[key] = float(m.group(1))
                except ValueError:
                    pass

        if len(scores) >= 3:
            for key in ["LCS", "ESS", "GRS", "COMP", "BIAS", "confidence"]:
                scores.setdefault(key, 0.5)
            if not self._is_template_echo(scores):
                return scores

        return dict(self._NEUTRAL)

    def compute_reward(self, claim, reasoning, evidence):
        if not reasoning.strip():
            return 0.0, {"LCS": 0.0, "ESS": 0.0, "GRS": 0.5, "COMP": 0.0, "BIAS": 0.5, "confidence": 0.0}

        if self.cache_scores:
            key = self._cache_key(claim, reasoning, evidence)
            if key in self._cache:
                scores = self._cache[key]
                return self._scores_to_reward(scores), scores

        prompt = self.build_prompt(claim, reasoning, evidence)
        try:
            if hasattr(self.llm, "generate_structured"):
                response, _ = self.llm.generate_structured(prompt)
            else:
                response, _ = self.llm.generate(prompt)
        except Exception as exc:
            _logger.warning(
                "LLMJudge: generation failed (%s) | model=%s | claim='%.80s' — "
                "returning neutral fallback scores.",
                exc,
                getattr(self.llm, "model_name", "unknown"),
                claim,
            )
            scores = {"LCS": 0.5, "ESS": 0.5, "GRS": 0.5, "COMP": 0.5, "BIAS": 0.5, "confidence": 0.0}
            # not cached — a transient failure shouldn't permanently deny
            # this (claim, reasoning, evidence) combo a real judge score
            return self._scores_to_reward(scores), scores

        scores = self.parse(response)
        if self.cache_scores:
            self._cache[key] = scores

        return self._scores_to_reward(scores), scores

    def _scores_to_reward(self, scores):
        lcs = float(scores.get("LCS", 0.5))
        ess = float(scores.get("ESS", 0.5))
        grs = float(scores.get("GRS", 0.5))
        comp = float(scores.get("COMP", 0.5))
        bias = float(scores.get("BIAS", 0.5))
        conf = float(scores.get("confidence", 0.5))

        if lcs == ess == grs == comp == bias == 0.5:
            # fallback/parse-failure sentinel — no usable judge signal, so
            # treat it as neutral rather than running it through the asymmetric
            # weight formula below (which maps all-0.5 to 0.175, not 0.5)
            return 0.5

        reward = (
            0.30 * lcs
            + 0.25 * ess
            + 0.20 * comp
            - 0.25 * grs
            - 0.15 * bias
        )
        # continuous blend toward neutral as the judge's own confidence drops,
        # instead of a hard threshold that made reward discontinuous in conf
        reward = conf * reward + (1 - conf) * 0.5
        return float(np.clip(reward, 0.0, 1.0))
