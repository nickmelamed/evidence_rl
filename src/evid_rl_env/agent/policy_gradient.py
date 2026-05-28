import numpy as np


class PolicyGradient:
    def __init__(self, policy, config):
        self.policy = policy
        self.lr = config.lr

    def update(self, trajectories):
        rewards = [r for (_, _, r) in trajectories]

        mean = np.mean(rewards)
        std = np.std(rewards) + 1e-8

        for state, action_idx, reward in trajectories:

            norm_r = (reward - mean) / std

            grad = self.policy.grad_log_prob(state, action_idx)

            self.policy.actor_params += self.lr * norm_r * grad