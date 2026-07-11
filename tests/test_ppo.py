import numpy as np
import pytest

from evid_rl_env.agent.ppo import PPO
from evid_rl_env.environment.state import State


class _FakeConfig:
    def __init__(self):
        self.clip = 0.2
        self.lr = 0.01
        self.gamma = 0.99
        self.entropy_coef = 0.01
        self.value_coef = 0.5
        self.max_grad_norm = 0.5
        self.ppo_epochs = 2
        self.gae_lambda = 0.95


def test_compute_advantages_single_step_matches_td_delta():
    ppo = PPO.__new__(PPO)
    ppo.gamma = 0.9
    ppo.gae_lambda = 0.95
    returns, advantages = ppo.compute_advantages([5.0], [2.0], next_value=0.0)
    assert returns[0] == pytest.approx(5.0)
    # a single-element z-score normalization always collapses to 0
    assert advantages[0] == pytest.approx(0.0)


def test_compute_advantages_earlier_steps_have_larger_raw_return():
    ppo = PPO.__new__(PPO)
    ppo.gamma = 0.99
    ppo.gae_lambda = 0.95
    returns, _ = ppo.compute_advantages([1.0, 1.0, 1.0], [0.0, 0.0, 0.0], next_value=0.0)
    assert returns[0] > returns[1] > returns[2]


def test_update_runs_and_changes_actor_and_value_params(fake_policy, monkeypatch):
    policy = fake_policy(state_dim=4)
    monkeypatch.setattr("evid_rl_env.agent.ppo.encode_state", lambda state: policy._features(state))

    ppo = PPO(policy, _FakeConfig())
    states = [State(claim="c", evidence_pool=[], steps_taken=i) for i in range(3)]
    trajectories = [
        (states[0], 0, 0.3, 1.0, 0.0),
        (states[1], 1, 0.2, -0.5, 0.0),
        (states[2], 2, 0.1, 2.0, 0.0),
    ]

    actor_before = policy.actor_params.copy()
    ppo.update(trajectories)

    assert not np.array_equal(actor_before, policy.actor_params)
    assert np.all(np.isfinite(policy.actor_params))
    assert np.all(np.isfinite(policy.value_params))


def test_gradient_clipping_bounds_extreme_advantage(fake_policy, monkeypatch):
    policy = fake_policy(state_dim=4)
    monkeypatch.setattr("evid_rl_env.agent.ppo.encode_state", lambda state: policy._features(state))

    config = _FakeConfig()
    config.max_grad_norm = 0.1
    ppo = PPO(policy, config)

    state = State(claim="c", evidence_pool=[], steps_taken=5)
    trajectories = [(state, 0, 0.5, 1e6, 0.0)]  # absurdly large reward/advantage

    ppo.update(trajectories)
    assert np.all(np.isfinite(policy.actor_params))
    assert np.all(np.isfinite(policy.value_params))
