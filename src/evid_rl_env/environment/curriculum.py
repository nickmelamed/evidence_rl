import random
from collections import deque


class Curriculum:
    def __init__(self, window=10, advance_threshold=0.75, retreat_threshold=0.4):
        self.level = 0  # 0=easy, 1=medium, 2=hard
        self.window = window
        self.advance_threshold = advance_threshold
        self.retreat_threshold = retreat_threshold
        self._history = deque(maxlen=window)

    def record(self, performance: float):
        """Call once per episode with a normalised performance score (0-1)."""
        self._history.append(performance)

    def update(self, performance: float):
        """Backwards-compatible single-call update."""
        self.record(performance)
        if len(self._history) < self.window:
            return
        mean = sum(self._history) / len(self._history)
        if mean > self.advance_threshold:
            self.level = min(2, self.level + 1)
            self._history.clear()
        elif mean < self.retreat_threshold:
            self.level = max(0, self.level - 1)
            self._history.clear()

    def sample(self, dataset):
        weights = []
        for d in dataset:
            diff_level = {"easy": 0, "medium": 1, "hard": 2}[d["difficulty"]]
            weight = 1.0 / (1 + abs(diff_level - self.level))
            weights.append(weight)
        return random.choices(dataset, weights=weights, k=1)[0]

    def _match_difficulty(self, diff):
        mapping = {"easy": 0, "medium": 1, "hard": 2}
        return mapping[diff] <= self.level
