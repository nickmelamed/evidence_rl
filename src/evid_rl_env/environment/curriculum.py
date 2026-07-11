import random
from collections import defaultdict, deque


class Curriculum:
    """Per-claim Prioritized Level Replay (Jiang et al. 2021).

    Each claim ("level") is scored individually, so sampling can concentrate
    on whichever specific claims the policy is actually still learning from,
    rather than relying on any a-priori difficulty label. A claim's score
    is:

      score = learning_progress + min_weight + staleness_coef * staleness

    - learning_progress = |short_mean - long_mean| of that claim's own
      task_success history (same idea as ALP-GMM's learning-progress signal,
      now per example instead of per bin). This is used instead of a
      value-loss/TD-error proxy so the same Curriculum works unmodified
      across PPO, plain policy gradient, and the bandit trainer, none of
      which expose a directly comparable value-error signal.
    - staleness = how long it's been since a claim was last sampled,
      normalised to [0, 1]. This is PLR's mechanism for preventing claims
      with a low historical score from being starved forever — it forces
      periodic revisits so a claim's score can be re-evaluated under the
      current policy.
    - min_weight is a constant floor so every claim keeps some sampling
      probability even at zero learning progress and zero staleness.

    Caveat: with hundreds of claims and a training run of comparable or
    fewer episodes, most claims are seen rarely, so early on the score is
    dominated by the cold-start weight and staleness bonus rather than a
    well-estimated learning-progress signal — the prioritization sharpens
    as more episodes accumulate per claim.
    """

    def __init__(self, short_window=5, long_window=20, min_weight=0.1, staleness_coef=0.1):
        self.short_window = short_window
        self.long_window = long_window
        self.min_weight = min_weight
        self.staleness_coef = staleness_coef
        self._short = defaultdict(lambda: deque(maxlen=short_window))
        self._long = defaultdict(lambda: deque(maxlen=long_window))
        self._last_seen_episode = {}
        self._episode = 0
        self._known_ids = []

    @staticmethod
    def _claim_key(claim: dict):
        """Claims are keyed by their 'id' field; falls back to the claim
        text itself if 'id' is absent (e.g. ad-hoc datasets in tests)."""
        return claim.get("id", claim.get("claim"))

    def record(self, claim_id, performance: float):
        """Call once per episode with the id of the claim that was just
        attempted and a normalised performance score (0-1)."""
        self._short[claim_id].append(performance)
        self._long[claim_id].append(performance)
        self._last_seen_episode[claim_id] = self._episode
        self._episode += 1

    def update(self, claim_id, performance: float):
        """Backwards-compatible single-call update."""
        self.record(claim_id, performance)

    def learning_progress(self, claim_id) -> float:
        long_ = self._long[claim_id]
        if not long_:
            return 1.0  # cold start: unseen claims get max learning-progress weight
        short = self._short[claim_id]
        short_mean = sum(short) / len(short)
        long_mean = sum(long_) / len(long_)
        return abs(short_mean - long_mean)

    def staleness(self, claim_id) -> float:
        last = self._last_seen_episode.get(claim_id)
        if last is None:
            return 1.0  # never sampled -> max staleness
        return min(1.0, (self._episode - last) / self.long_window)

    def score(self, claim_id) -> float:
        return (
            self.learning_progress(claim_id)
            + self.min_weight
            + self.staleness_coef * self.staleness(claim_id)
        )

    def sample(self, dataset):
        if not dataset:
            raise ValueError("Curriculum.sample() called with an empty dataset")
        self._known_ids = [self._claim_key(d) for d in dataset]
        weights = [self.score(i) for i in self._known_ids]
        total = sum(weights)
        # all-zero weights are reachable (e.g. min_weight=0 and every claim's
        # learning progress + staleness has simultaneously bottomed out) —
        # fall back to uniform rather than dividing by zero.
        probs = [w / total for w in weights] if total > 0 else None
        return random.choices(dataset, weights=probs, k=1)[0]

    @property
    def mean_score(self) -> float:
        """Read-only float for logging: the average current PLR score across
        every claim in the most recently sampled dataset. Falls toward the
        min_weight floor as the policy's performance stabilizes across the
        whole claim pool; stays elevated while the policy is still finding
        claims whose performance is shifting."""
        if not self._known_ids:
            return self.min_weight
        return sum(self.score(i) for i in self._known_ids) / len(self._known_ids)
