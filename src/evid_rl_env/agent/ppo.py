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

    def compute_advantages(self, rewards, values):
        advantages = []
        returns = []

        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = np.array(returns)
        values = np.array(values)

        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return returns, advantages

    def update(self, trajectories):
        states, actions, old_probs, rewards, values = zip(*trajectories)

        returns, advantages = self.compute_advantages(rewards, values)

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

            # updates
            self.policy.actor_params -= self.lr * actor_loss * grad_actor
            self.policy.value_params += self.lr * self.value_coef * grad_value