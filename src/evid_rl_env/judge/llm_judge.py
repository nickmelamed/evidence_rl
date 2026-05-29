import json
import re
import hashlib
import numpy as np


class LLMJudge:
    def __init__(self, llm, weight=0.5, cache_scores=True):
        self.llm = llm
        self.weight = weight
        self.cache_scores = cache_scores
        self._cache = {}

    def _cache_key(self, claim, reasoning, evidence):
        raw = claim + reasoning + "".join(e.text for e in evidence)
        return hashlib.md5(raw.encode()).hexdigest()

    def build_prompt(self, claim, reasoning, evidence):
        evidence_text = "\n".join([f"- {e.text}" for e in evidence])
        return f"""You are an expert evaluator of reasoning quality.

Evaluate the following:

CLAIM:
{claim}

EVIDENCE:
{evidence_text}

REASONING:
{reasoning}

Score the reasoning on these four dimensions.
Think step by step, then output ONLY a JSON object on the last line.

1. Logical Consistency (LCS): Is it coherent and non-contradictory? (0-1)
2. Evidence Support (ESS): Does it correctly use the evidence? (0-1)
3. Hallucination Risk (HRS): Does it invent unsupported facts? (0=safe, 1=risky)
4. Completeness (COMP): Does it fully address the claim? (0-1)

Output format (last line only, no markdown):
{{"LCS": 0.0, "ESS": 0.0, "HRS": 0.0, "COMP": 0.0, "confidence": 0.0}}"""

    def parse(self, response):
        # Try direct JSON parse first
        try:
            return json.loads(response.strip())
        except (json.JSONDecodeError, AttributeError):
            pass

        # Try extracting a JSON object with regex
        match = re.search(r'\{[^{}]*"LCS"[^{}]*\}', response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # Try extracting individual float values by key
        scores = {}
        for key in ["LCS", "ESS", "HRS", "COMP", "confidence"]:
            m = re.search(rf'"{key}"\s*:\s*([0-9.]+)', response)
            if m:
                try:
                    scores[key] = float(m.group(1))
                except ValueError:
                    pass

        if len(scores) >= 3:
            for key in ["LCS", "ESS", "HRS", "COMP", "confidence"]:
                scores.setdefault(key, 0.5)
            return scores

        # Hard fallback
        return {"LCS": 0.5, "ESS": 0.5, "HRS": 0.5, "COMP": 0.5, "confidence": 0.5}

    def compute_reward(self, claim, reasoning, evidence):
        if not reasoning.strip():
            return 0.0, {"LCS": 0.0, "ESS": 0.0, "HRS": 0.5, "COMP": 0.0, "confidence": 0.0}

        if self.cache_scores:
            key = self._cache_key(claim, reasoning, evidence)
            if key in self._cache:
                scores = self._cache[key]
                reward = self._scores_to_reward(scores)
                return reward, scores

        prompt = self.build_prompt(claim, reasoning, evidence)
        if hasattr(self.llm, "generate_structured"):
            response, _ = self.llm.generate_structured(prompt)
        else:
            response, _ = self.llm.generate(prompt)
        scores = self.parse(response)

        if self.cache_scores:
            self._cache[key] = scores

        reward = self._scores_to_reward(scores)
        return reward, scores

    def _scores_to_reward(self, scores):
        lcs = float(scores.get("LCS", 0.5))
        ess = float(scores.get("ESS", 0.5))
        hrs = float(scores.get("HRS", 0.5))
        comp = float(scores.get("COMP", 0.5))
        conf = float(scores.get("confidence", 0.5))

        reward = (0.35 * lcs + 0.35 * ess + 0.2 * comp - 0.3 * hrs) * conf
        return float(np.clip(reward, -1.0, 1.0))
