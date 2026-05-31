import numpy as np
from evid_rl_env.agent.policy import encode_state


class PolicyGradient:
    def __init__(self, policy, config):
        self.policy = policy
        self.lr = config.lr
        self.max_grad_norm = getattr(config, "max_grad_norm", 0.5)

    def update(self, trajectories):
        rewards = [r for (_, _, r) in trajectories]

        mean = np.mean(rewards)
        std = np.std(rewards) + 1e-8

        for state, action_idx, reward in trajectories:

            norm_r = (reward - mean) / std

            # Encode once and pass through so grad_log_prob doesn't re-encode.
            features = encode_state(state)
            grad = self.policy.grad_log_prob(state, action_idx, features=features)

            grad_norm = np.linalg.norm(grad)
            if grad_norm > self.max_grad_norm:
                grad = grad * (self.max_grad_norm / (grad_norm + 1e-8))

            self.policy.actor_params += self.lr * norm_r * grad