import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evid_rl_env.agent.config_loader import load_config
from evid_rl_env.agent.debate_trainer import DebateTrainer
from evid_rl_env.environment.actions import ACTIONS, Actions

DATASET = [{"claim": "Debate trainer claim.", "search_query": "debate trainer claim", "label": 1}]
_CONFIGS_DIR = Path(__file__).parent.parent / "configs"
SUPPORT_IDX = ACTIONS.index(Actions.SUPPORT)
CONTRADICT_IDX = ACTIONS.index(Actions.CONTRADICT)


class _IdentityNorm:
    def update(self, values):
        pass

    def normalize(self, value):
        return value


@pytest.fixture(autouse=True)
def _sandbox_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _trainer(fake_policy, make_env, episodes=2, seed=0, exp_name="test_debate"):
    _, config = load_config(str(_CONFIGS_DIR / "ppo_baseline.yaml"))
    return DebateTrainer(
        env=make_env(dataset=DATASET, seed=seed),
        support_policy=fake_policy(state_dim=4, seed=1),
        contradict_policy=fake_policy(state_dim=4, seed=2),
        config=config, episodes=episodes, exp_name=exp_name, seed=seed,
    )


def test_debate_trainer_runs_and_writes_one_checkpoint_per_role(fake_policy, make_env):
    trainer = _trainer(fake_policy, make_env)
    trainer.train()

    base = trainer.tracker.base_dir
    assert os.path.exists(os.path.join(base, "policy_support.npz"))
    assert os.path.exists(os.path.join(base, "policy_contradict.npz"))
    assert not os.path.exists(os.path.join(base, "policy.npz"))
    df = pd.read_csv(trainer.tracker.csv_path)
    assert len(df) == 2
    assert df["reward"].notna().all()
    assert df["reward_raw"].notna().all()


def test_each_role_is_updated_only_from_its_own_turns(fake_policy, make_env):
    trainer = _trainer(fake_policy, make_env, episodes=1)
    seen = {}
    for role in ("support", "contradict"):
        trainer.rl[role].update = lambda trajectories, role=role: seen.setdefault(role, trajectories)
    trainer.train()

    assert len(seen["support"]) == len(seen["contradict"]) == 5
    for features, mask, action_idx, *_ in seen["support"]:
        assert not mask[CONTRADICT_IDX]
        assert action_idx != CONTRADICT_IDX
    for features, mask, action_idx, *_ in seen["contradict"]:
        assert not mask[SUPPORT_IDX]
        assert action_idx != SUPPORT_IDX


def test_both_roles_receive_the_arbiters_final_reward(fake_policy, make_env):
    trainer = _trainer(fake_policy, make_env, episodes=1)
    trainer.reward_rms = _IdentityNorm()
    trainer.token_penalty = 0.0

    step_rewards = []
    infos = []
    original = trainer.debate.step

    def recording(action, payload):
        result = original(action, payload)
        step_rewards.append(result[1])
        infos.append(result[3])
        return result

    trainer.debate.step = recording
    seen = {}
    for role in ("support", "contradict"):
        trainer.rl[role].update = lambda trajectories, role=role: seen.setdefault(role, trajectories)
    trainer.train()

    final = infos[-1]["final_reward"]
    # trajectory tuples are (features, mask, action_idx, old_prob, reward, value)
    assert seen["support"][-1][4] == pytest.approx(step_rewards[-2] + final)
    assert seen["contradict"][-1][4] == pytest.approx(step_rewards[-1] + final)
    assert seen["support"][0][4] == pytest.approx(step_rewards[0])


def test_same_seed_gives_the_same_rewards_and_weights(fake_policy, make_env):
    first = _trainer(fake_policy, make_env, seed=4, exp_name="run_a")
    first.train()
    second = _trainer(fake_policy, make_env, seed=4, exp_name="run_b")
    second.train()

    a = pd.read_csv(first.tracker.csv_path)
    b = pd.read_csv(second.tracker.csv_path)
    np.testing.assert_array_equal(a["reward"], b["reward"])
    np.testing.assert_array_equal(a["reward_raw"], b["reward_raw"])
    for name in ("policy_support.npz", "policy_contradict.npz"):
        wa = np.load(os.path.join(first.tracker.base_dir, name))["actor_params"]
        wb = np.load(os.path.join(second.tracker.base_dir, name))["actor_params"]
        np.testing.assert_array_equal(wa, wb)


def test_the_run_config_records_the_roles_and_the_arbiter(fake_policy, make_env):
    trainer = _trainer(fake_policy, make_env, episodes=1)
    trainer.train()
    config = pd.io.json.read_json(os.path.join(trainer.tracker.base_dir, "config.json"),
                                  typ="series")
    assert config["turns_per_side"] == 5
    assert config["roles"] == ["support", "contradict"]
