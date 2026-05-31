import numpy as np
from evid_rl_env.agent.policy import encode_state


class PolicyGradient:
    def __init__(self, policy, config):
        self.policy = policy
        self.lr = config.lr
        self.max_grad_norm = getattr(config, "max_grad_norm", 0.5)
        self.lr_decay_episodes = getattr(config, "lr_decay_episodes", 200)
        self.lr_min_fraction = getattr(config, "lr_min_fraction", 0.2)
        self._episode_count = 0

    @property
    def current_lr(self) -> float:
        """LR after applying linear decay. Floors at lr_min_fraction * initial_lr."""
        if self.lr_decay_episodes <= 0:
            return self.lr
        fraction = max(
            self.lr_min_fraction,
            1.0 - (1.0 - self.lr_min_fraction) * self._episode_count / self.lr_decay_episodes,
        )
        return self.lr * fraction

    def update(self, trajectories):
        self._episode_count += 1
        effective_lr = self.current_lr

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

            self.policy.actor_params += effective_lr * norm_r * grad