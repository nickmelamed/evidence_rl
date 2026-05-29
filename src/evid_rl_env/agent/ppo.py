import numpy as np
from evid_rl_env.agent.policy import encode_state


class PPO:
    def __init__(self, policy, config):
        self.policy = policy
        self.clip = config.clip
        self.lr = config.lr
        self.gamma = config.gamma
        self.entropy_coef = config.entropy_coef
        self.value_coef = config.value_coef
        self.K = getattr(config, "ppo_epochs", 4)
        self.max_grad_norm = getattr(config, "max_grad_norm", 0.5)
        self.gae_lambda = getattr(config, "gae_lambda", 0.95)

    def compute_advantages(self, rewards, values, next_value=0.0):
        """
        Generalised Advantage Estimation (GAE-lambda).
        next_value: bootstrap value for the state after the last step (0 if terminal).
        """
        gae_lambda = getattr(self, "gae_lambda", 0.95)
        advantages = []
        gae = 0.0
        values_extended = list(values) + [next_value]

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values_extended[t + 1] - values_extended[t]
            gae = delta + self.gamma * gae_lambda * gae
            advantages.insert(0, gae)

        advantages = np.array(advantages, dtype=np.float32)
        returns = advantages + np.array(values, dtype=np.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages

    def update(self, trajectories):
        states, actions, old_probs, rewards, values = zip(*trajectories)

        returns, advantages = self.compute_advantages(list(rewards), list(values))

        for _ in range(self.K):
            for state, action_idx, old_prob, ret, adv in zip(
                states, actions, old_probs, returns, advantages
            ):
                probs = self.policy.get_probs(state)
                new_prob = probs[action_idx]

                ratio = new_prob / (old_prob + 1e-8)

                unclipped = ratio * adv
                clipped = np.clip(ratio, 1 - self.clip, 1 + self.clip) * adv

                actor_loss = -min(unclipped, clipped)

                # entropy bonus
                entropy = -np.sum(probs * np.log(probs + 1e-8))
                actor_loss -= self.entropy_coef * entropy

                # value loss
                value = self.policy.get_value(state)
                value_loss = (ret - value) ** 2

                # gradients
                grad_actor = self.policy.grad_log_prob(state, action_idx)
                features = encode_state(state)

                grad_value = features * (ret - value)

                # gradient clipping
                actor_norm = np.linalg.norm(grad_actor)
                if actor_norm > self.max_grad_norm:
                    grad_actor = grad_actor * (self.max_grad_norm / (actor_norm + 1e-8))

                value_norm = np.linalg.norm(grad_value)
                if value_norm > self.max_grad_norm:
                    grad_value = grad_value * (self.max_grad_norm / (value_norm + 1e-8))

                # updates
                self.policy.actor_params -= self.lr * actor_loss * grad_actor
                self.policy.value_params += self.lr * self.value_coef * grad_value