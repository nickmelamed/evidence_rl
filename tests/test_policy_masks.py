import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from evid_rl_env.agent import policy as policy_module
from evid_rl_env.agent.policy import ActorCriticPolicy, masked_softmax
from evid_rl_env.agent.policy_gradient import PolicyGradient
from evid_rl_env.agent.ppo import PPO
from evid_rl_env.environment.actions import ACTIONS, Actions
from evid_rl_env.environment.state import Evidence, State

STATE_DIM = 5


class _StubLLM:
    model_name = "stub"

    def generate(self, prompt):
        return "0.5", 1


def _policy(actor_params):
    p = ActorCriticPolicy.__new__(ActorCriticPolicy)
    p.actions = ACTIONS
    p.n_actions = len(ACTIONS)
    p.state_dim = STATE_DIM
    p.actor_params = actor_params
    p.value_params = np.zeros(STATE_DIM)
    p.llm = _StubLLM()
    return p


def _features_of(state):
    return np.array([
        len(state.selected_evidence), len(state.evidence_pool), state.query_count,
        0.0 if state.confidence is None else 1.0, state.steps_taken,
    ], dtype=np.float64)


@pytest.fixture(autouse=True)
def _simple_encoder(monkeypatch):
    monkeypatch.setattr(policy_module, "encode_state", _features_of)


def _state(n_selected=0, confidence=None, query_count=0):
    pool = [Evidence(id=i, text=f"t{i}", label="neutral") for i in range(3)]
    return State(claim="c", evidence_pool=pool, selected_evidence=pool[:n_selected],
                 confidence=confidence, query_count=query_count)


def _allowed_names(mask):
    return {a for a, ok in zip(ACTIONS, mask) if ok}


def test_masked_softmax_hand_example():
    probs = masked_softmax(np.array([0.0, 0.0, 5.0]), np.array([True, True, False]))
    np.testing.assert_allclose(probs, [0.5, 0.5, 0.0])


def test_mask_with_no_evidence_no_confidence_and_spent_queries():
    p = _policy(np.zeros((STATE_DIM, len(ACTIONS))))
    mask = p.action_mask(_state(n_selected=0, confidence=None, query_count=2))
    assert _allowed_names(mask) == {Actions.SELECT, Actions.RERANK}


def test_mask_with_evidence_and_confidence_leaves_everything_but_spent_queries():
    p = _policy(np.zeros((STATE_DIM, len(ACTIONS))))
    mask = p.action_mask(_state(n_selected=1, confidence=0.7, query_count=2))
    assert _allowed_names(mask) == set(ACTIONS) - {Actions.QUERY, Actions.REQUEST_CLARIFICATION}


@settings(max_examples=60, deadline=None)
@given(
    params=st.lists(st.floats(-3, 3), min_size=STATE_DIM * len(ACTIONS),
                    max_size=STATE_DIM * len(ACTIONS)),
    n_selected=st.integers(0, 3),
    confidence=st.one_of(st.none(), st.floats(0, 1)),
    query_count=st.integers(0, 2),
    seed=st.integers(0, 10_000),
)
def test_act_samples_from_the_masked_distribution_the_updater_will_use(
    params, n_selected, confidence, query_count, seed
):
    p = _policy(np.array(params).reshape(STATE_DIM, len(ACTIONS)))
    state = _state(n_selected, confidence, query_count)
    np.random.seed(seed)
    _, _, idx = p.act(state)

    mask = p.action_mask(state)
    assert mask[idx]
    np.testing.assert_array_equal(p.last_mask, mask)
    expected = masked_softmax(_features_of(state) @ p.actor_params, mask)
    np.testing.assert_allclose(p.last_probs, expected)
    assert np.all(p.last_probs[~mask] == 0.0)
    assert p.last_probs.sum() == pytest.approx(1.0)


class _Config:
    clip = 0.2
    lr = 0.01
    gamma = 0.99
    entropy_coef = 0.01
    value_coef = 0.5
    max_grad_norm = 10.0
    ppo_epochs = 1
    gae_lambda = 0.95


class _TinyPolicy:
    """Three actions and one feature, so the update can be worked by hand."""

    def __init__(self):
        self.actor_params = np.zeros((1, 3))
        self.value_params = np.zeros(1)

    def grad_log_prob(self, state, action_idx, features=None, probs=None):
        grad = -probs.copy()
        grad[action_idx] += 1
        return np.outer(features, grad)


def test_ppo_update_uses_the_masked_probabilities():
    # Masked probs are [.5, .5, 0]. A lone step has zero normalized advantage, so
    # the actor loss is only the entropy term, -0.01 * ln 2, and the step is
    # -lr * loss * grad with grad = [.5, -.5, 0]. An unmasked softmax would give
    # ln 3 and [2/3, -1/3, -1/3] instead.
    policy = _TinyPolicy()
    ppo = PPO(policy, _Config())
    mask = np.array([True, True, False])
    ppo.update([(np.array([1.0]), mask, 0, 0.5, 1.0, 0.0)])

    step = 0.01 * 0.01 * np.log(2)
    np.testing.assert_allclose(policy.actor_params, [[0.5 * step, -0.5 * step, 0.0]], atol=1e-9)


def test_policy_gradient_uses_the_masked_probabilities():
    policy = _TinyPolicy()

    class _Cfg:
        lr = 0.1
        max_grad_norm = 10.0

    pg = PolicyGradient(policy, _Cfg())
    mask = np.array([True, True, False])
    features = np.array([1.0])
    pg.update([(features, mask, 0, 1.0), (features, mask, 1, -1.0)])
    # Normalized rewards are +1 and -1. Step one pushes action 0 up by lr * [.5, -.5, 0].
    # Step two sees the updated params, so its probabilities are the masked softmax of
    # [a, -a, -] and it pushes action 1 down. The masked-out action never moves.
    lr = 0.1 * (1 - 0.8 / 200)
    a = lr * 0.5
    p0 = np.exp(a) / (np.exp(a) + np.exp(-a))
    assert policy.actor_params[0, 2] == 0.0
    assert policy.actor_params[0, 0] == pytest.approx(a + lr * p0, rel=1e-6)
