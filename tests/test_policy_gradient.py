import numpy as np
import pytest

from evid_rl_env.agent.policy_gradient import PolicyGradient
from evid_rl_env.environment.state import State


class _FakeConfig:
    def __init__(self):
        self.lr = 0.1
        self.max_grad_norm = 0.5
        self.lr_decay_episodes = 10
        self.lr_min_fraction = 0.2


def test_current_lr_starts_at_full_value():
    pg = PolicyGradient.__new__(PolicyGradient)
    pg.lr = 0.1
    pg.lr_decay_episodes = 10
    pg.lr_min_fraction = 0.2
    pg._episode_count = 0
    assert pg.current_lr == pytest.approx(0.1)


def test_current_lr_decays_linearly_then_floors():
    pg = PolicyGradient.__new__(PolicyGradient)
    pg.lr = 0.1
    pg.lr_decay_episodes = 10
    pg.lr_min_fraction = 0.2

    pg._episode_count = 5
    assert pg.current_lr == pytest.approx(0.06)  # 0.1 * (1 - 0.8*5/10)

    pg._episode_count = 1000  # far past the decay horizon
    assert pg.current_lr == pytest.approx(0.02)  # floors at lr * lr_min_fraction


def test_update_runs_and_changes_actor_params(fake_policy):
    policy = fake_policy(state_dim=4)

    pg = PolicyGradient(policy, _FakeConfig())
    features = [
        policy._features(State(claim="c", evidence_pool=[], steps_taken=i)) for i in range(3)
    ]
    mask = np.ones(policy.n_actions, dtype=bool)
    trajectories = [
        (features[0], mask, 0, 1.0),
        (features[1], mask, 1, -1.0),
        (features[2], mask, 2, 2.0),
    ]

    actor_before = policy.actor_params.copy()
    pg.update(trajectories)

    assert not np.array_equal(actor_before, policy.actor_params)
    assert np.all(np.isfinite(policy.actor_params))
    assert pg._episode_count == 1


def test_gradient_clipping_bounds_extreme_reward(fake_policy):
    policy = fake_policy(state_dim=4)
    config = _FakeConfig()
    config.max_grad_norm = 0.1
    pg = PolicyGradient(policy, config)

    features = policy._features(State(claim="c", evidence_pool=[], steps_taken=0))
    mask = np.ones(policy.n_actions, dtype=bool)
    trajectories = [(features, mask, 0, 1e6), (features, mask, 1, -1e6)]

    pg.update(trajectories)
    assert np.all(np.isfinite(policy.actor_params))
