from evid_rl_env.judge.reward import RewardFunction


class Judge:
    def __init__(self):
        self.reward_fn = RewardFunction()

    def evaluate(self, state):
        # mock final output (agent would generate this)
        final_output = {
            "confidence": 0.7,
            "true_score": 0.7,
            "reasoning": "Based on evidence..."
        }

        return self.reward_fn.compute(state, final_output)