import numpy as np


class RunningMeanStd:
    """Welford online algorithm for running mean and variance."""

    def __init__(self, epsilon=1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, x):
        x = np.asarray(x, dtype=np.float64).flatten()
        batch_mean = x.mean()
        batch_var = x.var()
        batch_count = len(x)
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        self.mean += delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot
        self.var = m2 / tot
        self.count = tot

    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)
