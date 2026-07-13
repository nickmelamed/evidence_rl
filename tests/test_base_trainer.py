"""
Unit test for BaseTrainer._init_common's eval_env construction
(agent/base_trainer.py) — confirms the run's seed is actually passed
through to the eval ClaimEnv instead of silently defaulting, with every
expensive dependency (ExperimentTracker's real directory creation,
ClaimEnv's real model loads) mocked out.
"""

import evid_rl_env.agent.base_trainer as base_trainer_module
from evid_rl_env.agent.base_trainer import BaseTrainer


def test_init_common_passes_seed_to_eval_env(monkeypatch, tmp_path):
    class _FakeExperimentTracker:
        def __init__(self, exp_name):
            self.base_dir = str(tmp_path / exp_name)

        def save_config(self, config):
            pass

    monkeypatch.setattr(base_trainer_module, "ExperimentTracker", _FakeExperimentTracker)

    captured = {}

    class _FakeClaimEnv:
        def __init__(self, dataset, **kwargs):
            captured["kwargs"] = kwargs

    monkeypatch.setattr("evid_rl_env.environment.environment.ClaimEnv", _FakeClaimEnv)

    fake_env = type("_FakeEnv", (), {"dataset": []})()
    fake_policy = type("_FakePolicy", (), {"llm": None})()  # no llm -> skips build_standard_baselines
    fake_config = type("_FakeConfig", (), {"gold_judge_model": None})()  # skips gold evaluator

    trainer = BaseTrainer()
    trainer._init_common(
        env=fake_env, policy=fake_policy, config=fake_config, episodes=1,
        exp_name="test_seed_passthrough", seed=99, use_wandb=False,
        eval_dataset=[{"claim": "x"}], eval_every=1, baseline_n_episodes=1,
        curriculum=None,
    )

    assert captured["kwargs"]["seed"] == 99
