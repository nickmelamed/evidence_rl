import csv
import random

import numpy as np

from evid_rl_env.agent.eval_mixin import _CSV_COLUMNS, EvalMixin


class _Dummy(EvalMixin):
    pass


def test_append_eval_csv_writes_all_columns_including_bias(tmp_path):
    dummy = _Dummy()
    dummy._eval_csv_path = str(tmp_path / "eval_metrics.csv")

    rl_metrics = {
        "eval/llm_LCS": 0.7,
        "eval/llm_ESS": 0.6,
        "eval/llm_GRS": 0.2,
        "eval/llm_COMP": 0.8,
        "eval/llm_BIAS": 0.3,
    }
    baseline_results = {"greedy_llm": {"mean_reward": 0.5}}

    dummy._append_eval_csv(0, 0.6, 0.1, baseline_results, 0.5, rl_metrics)

    with open(dummy._eval_csv_path) as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    assert set(rows[0].keys()) == set(_CSV_COLUMNS)
    assert rows[0]["bias"] == "0.3"
    assert rows[0]["delta_vs_greedy_llm"] == str(0.6 - 0.5)


class _StubEvaluator:
    def evaluate(self):
        return {
            "eval/mean_reward_raw": 0.5,
            "eval/std_reward_raw": 0.1,
            "eval/mean_reward": 0.5,
            "eval/std_reward": 0.1,
            "eval/llm_LCS": 0.5,
            "eval/llm_ESS": 0.5,
            "eval/llm_GRS": 0.5,
            "eval/llm_COMP": 0.5,
            "eval/llm_BIAS": 0.5,
        }


class _StubTracker:
    def log_eval(self, *args, **kwargs):
        pass


class _StubBaseline:
    def run(self, n_episodes):
        # Deliberately consumes RNG entropy, to prove _run_eval_round isolates it.
        random.random()
        np.random.random()
        return {"mean_reward": 0.4, "std_reward": 0.05}


class _CountingStubBaseline:
    def __init__(self, mean_reward=0.4):
        self.call_count = 0
        self._mean_reward = mean_reward

    def run(self, n_episodes):
        self.call_count += 1
        return {"mean_reward": self._mean_reward, "std_reward": 0.05}


def test_run_eval_round_does_not_leak_rng_state(tmp_path):
    dummy = _Dummy()
    dummy.evaluator = _StubEvaluator()
    dummy.baselines = {"greedy_llm": _StubBaseline()}
    dummy.baseline_n_episodes = 1
    dummy.seed = 42
    dummy.use_wandb = False
    dummy.tracker = _StubTracker()
    dummy._eval_csv_path = str(tmp_path / "eval_metrics.csv")

    random.seed(999)
    np.random.seed(999)
    py_state_before = random.getstate()
    np_state_before = np.random.get_state()

    dummy._run_eval_round(0)

    assert random.getstate() == py_state_before
    for a, b in zip(np.random.get_state(), np_state_before):
        if isinstance(a, np.ndarray):
            assert np.array_equal(a, b)
        else:
            assert a == b


def _make_dummy_with_baselines(tmp_path, cheap, expensive, expensive_baseline_every=5):
    dummy = _Dummy()
    dummy.evaluator = _StubEvaluator()
    dummy.baselines = {"greedy_llm": cheap, "fewshot_k3": expensive}
    dummy.baseline_n_episodes = 1
    dummy.seed = 42
    dummy.use_wandb = False
    dummy.tracker = _StubTracker()
    dummy._eval_csv_path = str(tmp_path / "eval_metrics.csv")
    dummy.expensive_baseline_every = expensive_baseline_every
    return dummy


def test_expensive_baseline_runs_only_on_cadence_round(tmp_path):
    cheap = _CountingStubBaseline()
    expensive = _CountingStubBaseline()
    dummy = _make_dummy_with_baselines(tmp_path, cheap, expensive, expensive_baseline_every=5)

    for ep in range(5):
        dummy._run_eval_round(ep)

    assert cheap.call_count == 5
    assert expensive.call_count == 1


def test_expensive_baseline_every_one_runs_every_round(tmp_path):
    """expensive_baseline_every=1 restores today's every-round behavior."""
    cheap = _CountingStubBaseline()
    expensive = _CountingStubBaseline()
    dummy = _make_dummy_with_baselines(tmp_path, cheap, expensive, expensive_baseline_every=1)

    for ep in range(3):
        dummy._run_eval_round(ep)

    assert cheap.call_count == 3
    assert expensive.call_count == 3


def test_skipped_round_still_produces_valid_csv_row_and_ref(tmp_path):
    cheap = _CountingStubBaseline(mean_reward=0.3)
    expensive = _CountingStubBaseline()
    dummy = _make_dummy_with_baselines(tmp_path, cheap, expensive, expensive_baseline_every=5)

    dummy._run_eval_round(0)  # round 1 — expensive baseline skipped

    with open(dummy._eval_csv_path) as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    assert rows[0]["greedy_llm_mean"] == "0.3"
    assert rows[0]["fewshot_k3_mean"] == ""
    # ref (delta_vs_greedy_llm) must still be computed off greedy_llm, which
    # always runs regardless of the expensive-baseline cadence
    assert rows[0]["delta_vs_greedy_llm"] == str(0.5 - 0.3)
