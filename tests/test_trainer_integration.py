import os
from pathlib import Path

import pandas as pd
import pytest

from evid_rl_env.agent.bandit_trainer import BanditTrainer
from evid_rl_env.agent.config_loader import load_config
from evid_rl_env.agent.trainer import Trainer
from evid_rl_env.environment.curriculum import Curriculum

DATASET = [{"claim": "Integration test claim.", "search_query": "integration test claim"}]

# Absolute path so this still resolves after _sandbox_cwd below chdir's into a
# tmp_path — configs/*_baseline.yaml is the single source of truth for RL
# hyperparameters (PPOConfig()/PGConfig()/BanditConfig() alone leave the
# tuning fields as None), so these tests load the real config files rather
# than bare-constructing a Config class.
_CONFIGS_DIR = Path(__file__).parent.parent / "configs"


@pytest.fixture(autouse=True)
def _sandbox_cwd(tmp_path, monkeypatch):
    """Trainer/BanditTrainer write metrics.csv/config.json/policy.npz under
    artifacts/experiments/<run> relative to cwd — sandbox it per test so the
    suite never touches the real repo's artifacts/ directory."""
    monkeypatch.chdir(tmp_path)


def _metrics_df(trainer):
    return pd.read_csv(trainer.tracker.csv_path)


def test_trainer_ppo_runs_end_to_end(fake_policy, make_env, monkeypatch):
    policy = fake_policy(state_dim=4)
    monkeypatch.setattr("evid_rl_env.agent.ppo.encode_state", lambda state: policy._features(state))

    _, config = load_config(str(_CONFIGS_DIR / "ppo_baseline.yaml"))
    env = make_env(dataset=DATASET)
    trainer = Trainer(
        env=env, policy=policy, config=config, episodes=2, algo="ppo",
        exp_name="test_ppo", eval_dataset=None, curriculum=None,
    )
    trainer.train()

    checkpoint_path = os.path.join(trainer.tracker.base_dir, "policy.npz")
    assert os.path.exists(checkpoint_path)

    df = _metrics_df(trainer)
    assert len(df) == 2
    assert df["reward"].notna().all()
    assert df["reward_raw"].notna().all()  # previously dropped by FIXED_FIELDS


def test_trainer_pg_runs_end_to_end_and_persists_pg_lr(fake_policy, make_env, monkeypatch):
    policy = fake_policy(state_dim=4)
    monkeypatch.setattr(
        "evid_rl_env.agent.policy_gradient.encode_state", lambda state: policy._features(state)
    )

    _, config = load_config(str(_CONFIGS_DIR / "pg_baseline.yaml"))
    env = make_env(dataset=DATASET)
    trainer = Trainer(
        env=env, policy=policy, config=config, episodes=2, algo="pg",
        exp_name="test_pg", eval_dataset=None, curriculum=None,
    )
    trainer.train()

    df = _metrics_df(trainer)
    assert len(df) == 2
    assert df["pg_lr"].notna().all()  # previously dropped by FIXED_FIELDS


def test_trainer_with_curriculum_persists_curriculum_mean_score(fake_policy, make_env, monkeypatch):
    policy = fake_policy(state_dim=4)
    monkeypatch.setattr("evid_rl_env.agent.ppo.encode_state", lambda state: policy._features(state))

    _, config = load_config(str(_CONFIGS_DIR / "ppo_baseline.yaml"))
    env = make_env(dataset=DATASET)
    trainer = Trainer(
        env=env, policy=policy, config=config, episodes=2, algo="ppo",
        exp_name="test_curriculum", eval_dataset=None, curriculum=Curriculum(short_window=3),
    )
    trainer.train()

    df = _metrics_df(trainer)
    assert df["curriculum_mean_score"].notna().all()  # previously dropped by FIXED_FIELDS


def test_bandit_trainer_runs_end_to_end_and_populates_action_dist(fake_policy, make_env, monkeypatch):
    """Regression test: BanditTrainer used to never flatten action_dist into
    the dotted columns FIXED_FIELDS expects, so every bandit run's
    action-distribution columns in metrics.csv were silently empty."""
    policy = fake_policy(state_dim=4)
    monkeypatch.setattr(
        "evid_rl_env.agent.bandit_trainer.encode_state", lambda state: policy._features(state)
    )

    _, config = load_config(str(_CONFIGS_DIR / "bandit_baseline.yaml"))
    env = make_env(dataset=DATASET)
    trainer = BanditTrainer(
        env=env, policy=policy, config=config, episodes=2,
        exp_name="test_bandit", eval_dataset=None, curriculum=None,
    )
    trainer.train()

    checkpoint_path = os.path.join(trainer.tracker.base_dir, "policy.npz")
    assert os.path.exists(checkpoint_path)

    df = _metrics_df(trainer)
    assert len(df) == 2

    action_dist_cols = [c for c in df.columns if c.startswith("action_dist.")]
    assert df[action_dist_cols].notna().any().any()  # at least one populated cell

    assert df["reward"].notna().all()
    assert df["reward_raw"].notna().all()
