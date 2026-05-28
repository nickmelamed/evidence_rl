import json
import numpy as np


class LLMJudge:
    def __init__(self, llm, weight=0.5):
        """
        weight: how much LLM reward contributes vs base reward
        """
        self.llm = llm
        self.weight = weight

    def build_prompt(self, claim, reasoning, evidence):
        evidence_text = "\n".join([f"- {e.text}" for e in evidence])

        return f"""
You are an expert evaluator of reasoning quality.

Evaluate the following:

CLAIM:
{claim}

EVIDENCE:
{evidence_text}

REASONING:
{reasoning}

Score the reasoning on:

1. Logical Consistency (LCS): Is it coherent and non-contradictory?
2. Evidence Support (ESS): Does it correctly use evidence?
3. Hallucination Risk (HRS): Does it invent unsupported facts? (higher = worse)
4. Completeness (COMP): Does it fully address the claim?

Return ONLY JSON:
{{
  "LCS": float (0-1),
  "ESS": float (0-1),
  "HRS": float (0-1),
  "COMP": float (0-1),
  "confidence": float (0-1)
}}
"""

    def parse(self, response):
        try:
            data = json.loads(response)
            return data
        except:
            # fallback
            return {
                "LCS": 0.5,
                "ESS": 0.5,
                "HRS": 0.5,
                "COMP": 0.5,
                "confidence": 0.5
            }

    def compute_reward(self, claim, reasoning, evidence):
        prompt = self.build_prompt(claim, reasoning, evidence)
        response = self.llm.generate(prompt)

        scores = self.parse(response)

        # RLHF-style reward shaping
        lcs = scores["LCS"]
        ess = scores["ESS"]
        hrs = scores["HRS"]
        comp = scores["COMP"]
        conf = scores["confidence"]

        # hallucination is penalized
        reward = (
            0.35 * lcs +
            0.35 * ess +
            0.2 * comp -
            0.3 * hrs
        )

        # confidence scaling
        reward *= conf

        # normalize to [-1, 1]
        reward = float(np.clip(reward, -1.0, 1.0))

        return reward, scores