import numpy as np

class LinUCBBandit:
    def __init__(self, n_actions, d, alpha=1.0):
        self.A = [np.eye(d) for _ in range(n_actions)]
        self.b = [np.zeros(d) for _ in range(n_actions)]
        self.alpha = alpha

    def select_action(self, x):
        p = []
        for a in range(len(self.A)):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            p.append(theta @ x + self.alpha * np.sqrt(x @ A_inv @ x))
        return np.argmax(p)

    def update(self, action, x, reward):
        self.A[action] += np.outer(x, x)
        self.b[action] += reward * x