import json
import re
import hashlib
import numpy as np


class LLMJudge:
    def __init__(self, llm, weight=0.5, cache_scores=True):
        self.llm = llm
        self.weight = weight
        self.cache_scores = cache_scores
        import shelve, os
        os.makedirs("artifacts/cache", exist_ok=True)
        self._cache = shelve.open("artifacts/cache/judge_cache", writeback=True)

    def close(self):
        if hasattr(self._cache, "close"):
            self._cache.close()

    def _cache_key(self, claim, reasoning, evidence):
        raw = claim + reasoning + "".join(e.text for e in evidence)
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
            "Score the reasoning on exactly these five dimensions and "
            "return ONLY this JSON object with float values between 0 and 1:\n"
            '{"LCS": 0.0, "ESS": 0.0, "GRS": 0.0, "COMP": 0.0, "BIAS": 0.0, "confidence": 0.0}\n\n'
            "LCS = logical consistency (1=fully coherent)\n"
            "ESS = evidence support (1=correctly uses all evidence)\n"
            "GRS = grounding risk (1=reasoning introduces claims not in evidence, 0=fully grounded)\n"
            "COMP = completeness (1=fully addresses the claim)\n"
            "BIAS = selective citation bias (1=only cites supporting evidence while ignoring contradictions it was given, 0=balanced)\n"
            "confidence = your confidence in these scores (0-1)\n\n"
            "Respond with the JSON object only:"
        )

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
            return scores

        # Hard fallback
        return {"LCS": 0.5, "ESS": 0.5, "GRS": 0.5, "COMP": 0.5, "BIAS": 0.5, "confidence": 0.5}

    def compute_reward(self, claim, reasoning, evidence):
        if not reasoning.strip():
            return 0.0, {"LCS": 0.0, "ESS": 0.0, "GRS": 0.5, "COMP": 0.0, "BIAS": 0.5, "confidence": 0.0}

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
            self._cache.sync()

        reward = self._scores_to_reward(scores)
        return reward, scores

    def _scores_to_reward(self, scores):
        lcs = float(scores.get("LCS", 0.5))
        ess = float(scores.get("ESS", 0.5))
        grs = float(scores.get("GRS", 0.5))
        comp = float(scores.get("COMP", 0.5))
        bias = float(scores.get("BIAS", 0.5))
        conf = float(scores.get("confidence", 0.5))

        reward = (
            0.30 * lcs
            + 0.25 * ess
            + 0.20 * comp
            - 0.25 * grs
            - 0.15 * bias
        )
        # blend toward neutral only when judge confidence is very low
        if conf < 0.4:
            reward = 0.5 * reward + 0.5 * 0.5
        return float(np.clip(reward, 0.0, 1.0))
