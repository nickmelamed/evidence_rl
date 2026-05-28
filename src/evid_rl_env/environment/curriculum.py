import random


class Curriculum:
    def __init__(self):
        self.level = 0  # 0=easy, 1=medium, 2=hard

    def sample(self, dataset):
        weights = []

        for d in dataset:
            diff_level = {"easy": 0, "medium": 1, "hard": 2}[d["difficulty"]]

            # bias toward current level
            weight = 1.0 / (1 + abs(diff_level - self.level))
            weights.append(weight)

        return random.choices(dataset, weights=weights, k=1)[0]

    def _match_difficulty(self, diff):
        mapping = {"easy": 0, "medium": 1, "hard": 2}
        return mapping[diff] <= self.level

    def update(self, performance):
        if performance > 0.8:
            self.level = min(2, self.level + 1)