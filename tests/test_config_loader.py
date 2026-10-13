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


def _training_judges(cfg, debate_models):
    training = {cfg.judge_model}
    training.update(getattr(cfg, "judge_ensemble_models", None) or [])
    if getattr(cfg, "judge_escalation", False) and cfg.judge_escalation_target == "debate":
        training |= debate_models
    if getattr(cfg, "arbiter_model", None):
        training.add(cfg.arbiter_model)
    return training


def test_gold_judge_is_never_one_of_the_training_judges():
    import glob

    from evid_rl_env.judge import debate_judge

    debate_models = {
        debate_judge._DEFAULT_ADVOCATE_FOR_MODEL,
        debate_judge._DEFAULT_ADVOCATE_AGAINST_MODEL,
        debate_judge._DEFAULT_ARBITER_MODEL,
    }
    paths = [p for p in sorted(glob.glob("configs/*.yaml")) if not p.endswith("base.yaml")]
    assert paths
    for path in paths:
        _, cfg = load_config(path)
        training = _training_judges(cfg, debate_models)
        assert cfg.gold_judge_model not in training, path


def test_multiagent_config_sets_the_arbiter_and_the_turn_count():
    algo, cfg = load_config("configs/ppo_multiagent_debate.yaml")
    assert algo == "ppo_multiagent"
    assert cfg.arbiter_model == "Qwen/Qwen2.5-3B-Instruct"
    assert cfg.turns_per_side == 5
    assert cfg.max_tokens_per_episode > 2000
    assert cfg.gold_judge_model != cfg.arbiter_model


def test_guard_catches_a_gold_judge_that_is_also_the_arbiter(tmp_path):
    (tmp_path / "base.yaml").write_text("seed: 1\n")
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "algo: ppo_multiagent\n"
        "rl:\n"
        "  arbiter_model: org/shared-model\n"
        "gold_judge_model: org/shared-model\n"
    )
    _, cfg = load_config(str(bad))
    assert cfg.gold_judge_model in _training_judges(cfg, debate_models=set())
