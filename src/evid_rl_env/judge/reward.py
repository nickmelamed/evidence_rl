from evid_rl_env.judge.metrics import (
    compute_ess,
    compute_ecs,
    compute_adversarial_penalty
)


class RewardFunction:
    def __init__(self):
        pass

    def compute(self, state, final_output):
        """
        Final_output must contain:
        - reasoning (str)
        - confidence (float)
        - true_score (float)
        """

        # pass reasoning text, not dict
        reasoning = final_output.get("reasoning", "")

        # core metrics
        ess = compute_ess(state.selected_evidence)
        ecs = compute_ecs(state.selected_evidence)

        adversarial_penalty = compute_adversarial_penalty(state.selected_evidence)

        # uncertainty calibration
        confidence = final_output.get("confidence", 0.5)
        true_score = final_output.get("true_score", 0.5)

        uncertainty_penalty = abs(confidence - true_score)

        # reward aggregation
        reward = (
            0.4 * ess
            + 0.3 * (1 - ecs)
            - 0.2 * adversarial_penalty
            - 0.2 * uncertainty_penalty
        )

        # anti-gaming penalties
        if len(state.selected_evidence) > 5:
            reward -= 0.2

        # Penalize no reasoning
        if len(reasoning.strip()) == 0:
            reward -= 0.1

        return max(0.0, min(1.0, reward))