from pathlib import Path

import numpy as np
import pytest

from evid_rl_env.agent.config_loader import load_config
from evid_rl_env.agent.trainer import Trainer
from evid_rl_env.environment.actions import ACTIONS, Actions

DATASET = [{"claim": "Snapshot test claim.", "search_query": "snapshot test claim"}]
_CONFIGS_DIR = Path(__file__).parent.parent / "configs"


@pytest.fixture(autouse=True)
def _sandbox_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


@pytest.mark.parametrize("algo,config_file", [("ppo", "ppo_baseline.yaml"),
                                              ("pg", "pg_baseline.yaml")])
def test_update_receives_the_features_each_action_was_chosen_from(
    fake_policy, make_env, algo, config_file
):
    """ClaimEnv mutates one State in place, so the trainer must snapshot what the
    policy saw at act time. Otherwise every step is updated with end-of-episode inputs."""
    np.random.seed(3)
    policy = fake_policy(state_dim=4)
    seen_at_act = []
    original_act = policy.act

    def recording_act(state, *args, **kwargs):
        seen_at_act.append(policy._features(state).copy())
        # SELECT every turn so the episode runs to the step limit
        return original_act(state, force_action_idx=ACTIONS.index(Actions.SELECT))

    policy.act = recording_act

    _, config = load_config(str(_CONFIGS_DIR / config_file))
    trainer = Trainer(
        env=make_env(dataset=DATASET), policy=policy, config=config, episodes=1,
        algo=algo, exp_name=f"test_snapshot_{algo}", eval_dataset=None, curriculum=None,
    )
    recorded = []
    trainer.rl.update = lambda trajectories: recorded.extend(trajectories)
    trainer.train()

    assert len(recorded) == len(seen_at_act) > 1
    for entry, expected in zip(recorded, seen_at_act):
        np.testing.assert_array_equal(entry[0], expected)
