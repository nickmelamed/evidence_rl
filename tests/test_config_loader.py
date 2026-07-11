import pytest

from evid_rl_env.agent.config import BanditConfig, PGConfig, PPOConfig
from evid_rl_env.agent.config_loader import load_base_config, load_config


def test_load_base_config_reads_seed_and_annotator():
    cfg = load_base_config("configs")
    assert cfg["seed"] == 42
    assert "default_annotator_model" in cfg


def test_load_base_config_missing_file_returns_empty_dict(tmp_path):
    assert load_base_config(str(tmp_path)) == {}


def test_load_config_ppo_merges_base_and_algo_overrides():
    algo, cfg = load_config("configs/ppo_baseline.yaml")
    assert algo == "ppo"
    assert isinstance(cfg, PPOConfig)
    assert cfg.seed == 42            # inherited from base.yaml, not overridden
    assert cfg.lr == 0.001           # from ppo_baseline.yaml
    assert cfg.clip == 0.2
    assert cfg.actor_model == "google/gemma-2-2b-it"


def test_load_config_pg_returns_pg_config():
    algo, cfg = load_config("configs/pg_baseline.yaml")
    assert algo == "pg"
    assert isinstance(cfg, PGConfig)


def test_load_config_bandit_returns_bandit_config():
    algo, cfg = load_config("configs/bandit_baseline.yaml")
    assert algo == "bandit"
    assert isinstance(cfg, BanditConfig)
    assert cfg.alpha == 1.0


def test_load_config_unknown_algo_raises(tmp_path):
    (tmp_path / "base.yaml").write_text("seed: 1\n")
    bogus = tmp_path / "bogus.yaml"
    bogus.write_text("algo: nonsense\n")
    with pytest.raises(ValueError):
        load_config(str(bogus))
