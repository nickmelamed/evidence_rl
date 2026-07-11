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
        return int(np.argmax(p))

    def update(self, action, x, reward):
        self.A[action] += np.outer(x, x)
        self.b[action] += reward * x

    def save(self, path: str, model_name: str = "google/gemma-2-2b-it"):
        np.savez(
            path,
            type=np.array(["bandit"]),
            A=np.stack(self.A),
            b=np.stack(self.b),
            alpha=np.array([self.alpha]),
            n_actions=np.array([len(self.A)]),
            d=np.array([self.A[0].shape[0]]),
            model_name=np.array([model_name], dtype="U256"),
        )

    @classmethod
    def load(cls, path: str):
        data = np.load(path, allow_pickle=False)
        n_actions = int(data["n_actions"][0])
        alpha = float(data["alpha"][0])
        d = int(data["d"][0])
        bandit = cls(n_actions=n_actions, d=d, alpha=alpha)
        bandit.A = [data["A"][i] for i in range(n_actions)]
        bandit.b = [data["b"][i] for i in range(n_actions)]
        return bandit, str(data["model_name"][0])