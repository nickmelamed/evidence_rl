from evid_rl_env.judge.metrics import (
    compute_adversarial_contamination,
    compute_contradiction_acknowledgment,
    compute_f1,
)


class RewardFunction:
    def __init__(
        self,
        w_f1=0.40,
        w_contradiction=0.20,
        w_adversarial=0.15,
        w_uncertainty=0.15,
        w_llm=0.10,
        evidence_budget=5,
        overselection_slope=0.04,
    ):
        self.w_f1 = w_f1
        self.w_contradiction = w_contradiction
        self.w_adversarial = w_adversarial
        self.w_uncertainty = w_uncertainty
        self.w_llm = w_llm
        self.evidence_budget = evidence_budget
        self.overselection_slope = overselection_slope

    def compute(self, state, final_output, llm_reward: float):
        reasoning = final_output.get("reasoning", "")

        if reasoning.strip() == "":
            return 0.0

        selected = state.selected_evidence
        available = state.evidence_pool

        f1 = compute_f1(selected, available)
        ca = compute_contradiction_acknowledgment(selected, available)
        ac = compute_adversarial_contamination(selected)

        confidence = final_output.get("confidence", 0.5)
        true_score = final_output.get("true_score", 0.5)
        uncertainty_penalty = abs(confidence - true_score)

        overselection_penalty = max(0, len(selected) - self.evidence_budget) * self.overselection_slope

        reward = (
            self.w_f1 * f1
            + self.w_contradiction * ca
            - self.w_adversarial * ac
            - self.w_uncertainty * uncertainty_penalty
            + self.w_llm * llm_reward
            - overselection_penalty
        )

        return float(max(-1.0, min(1.0, reward)))
