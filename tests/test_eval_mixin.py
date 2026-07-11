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
