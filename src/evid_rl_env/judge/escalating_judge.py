from evid_rl_env.judge.llm_judge import LLMJudge


class EscalatingJudge:
    """Cheap tier-1 judge for the common case; escalates to a tier-2
    EnsembleJudge only when the cheap judge's own signals suggest it might
    be wrong or gaming-prone — low self-reported confidence, high grounding-
    risk/bias scores, or adversarial-labeled evidence in the pool (already
    flagged by EvidenceLabeler on Evidence.label, so this check costs no
    extra model call). Same (claim, reasoning, evidence) -> (reward, scores)
    interface as LLMJudge/EnsembleJudge, so it's a drop-in wherever
    self.llm_judge.compute_reward is called (see environment.py).

    Note: trigger (1) still depends on the cheap judge's self-reported
    confidence, which isn't fully trustworthy on its own (see
    EnsembleJudge) — a sufficiently adversarial policy could in principle
    learn to keep that number high specifically to dodge escalation.
    Triggers (2) and (3) don't share that weakness, so this is still
    strictly more robust than a single judge, just not a complete fix for
    confidence-gaming by itself.
    """

    def __init__(
        self,
        cheap_judge: LLMJudge,
        escalated_judge,
        confidence_threshold: float = 0.5,
        grs_threshold: float = 0.5,
        bias_threshold: float = 0.5,
    ):
        self.cheap_judge = cheap_judge
        self.escalated_judge = escalated_judge
        self.confidence_threshold = confidence_threshold
        self.grs_threshold = grs_threshold
        self.bias_threshold = bias_threshold

    def _should_escalate(self, cheap_scores: dict, evidence: list) -> bool:
        if float(cheap_scores.get("confidence", 1.0)) < self.confidence_threshold:
            return True
        if float(cheap_scores.get("GRS", 0.0)) > self.grs_threshold:
            return True
        if float(cheap_scores.get("BIAS", 0.0)) > self.bias_threshold:
            return True
        if any(getattr(e, "label", None) == "adversarial" for e in evidence):
            return True
        return False

    def compute_reward(self, claim, reasoning, evidence):
        if not reasoning.strip():
            return 0.0, {
                "LCS": 0.0, "ESS": 0.0, "GRS": 0.5, "COMP": 0.0, "BIAS": 0.5,
                "confidence": 0.0, "escalated": False,
            }

        cheap_scores = self.cheap_judge.get_scores(claim, reasoning, evidence)

        if self._should_escalate(cheap_scores, evidence):
            reward, scores = self.escalated_judge.compute_reward(claim, reasoning, evidence)
            scores = dict(scores)
            scores["escalated"] = True
            return reward, scores

        scores = dict(cheap_scores)
        scores["escalated"] = False
        return LLMJudge._scores_to_reward(cheap_scores), scores
