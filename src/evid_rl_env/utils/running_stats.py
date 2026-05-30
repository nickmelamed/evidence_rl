import numpy as np


class RunningMeanStd:
    """Welford online algorithm for running mean and variance."""

    def __init__(self, epsilon=1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon
        # AUDIT FIX: lock flag prevents accidental updates inside eval/baseline loops;
        # set via lock()/unlock() context helpers in Evaluator
        self._locked = False

    def update(self, x):
        # AUDIT FIX: raise immediately if update() is called while locked so that
        # eval or baseline code paths never silently corrupt training normalizer state
        assert not self._locked, (
            "RunningMeanStd.update() called while locked — normalizer must not be "
            "updated inside an eval loop, baseline run, or trajectory collection. "
            "Only call update() from the training step."
        )
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
