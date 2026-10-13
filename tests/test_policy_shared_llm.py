import numpy as np

from evid_rl_env.agent import policy as policy_module
from evid_rl_env.agent.policy import ActorCriticPolicy


def test_a_policy_can_reuse_an_existing_llm_client(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a second client was built")

    monkeypatch.setattr(policy_module, "LLMClient", refuse)
    shared = object()
    first = ActorCriticPolicy(n_actions=13, state_dim=4, llm=shared)
    second = ActorCriticPolicy(n_actions=13, state_dim=4, llm=shared)
    assert first.llm is second.llm is shared


def test_a_policy_builds_its_own_client_by_default(monkeypatch):
    built = []

    class _Client:
        def __init__(self, model_name, seed):
            built.append((model_name, seed))

    monkeypatch.setattr(policy_module, "LLMClient", _Client)
    policy = ActorCriticPolicy(n_actions=13, state_dim=4, model_name="org/actor", seed=5)
    assert built == [("org/actor", 5)]
    assert isinstance(policy.llm, _Client)


def test_sharing_a_client_leaves_the_actor_weights_independent():
    shared = object()
    a = ActorCriticPolicy(n_actions=13, state_dim=4, llm=shared)
    b = ActorCriticPolicy(n_actions=13, state_dim=4, llm=shared)
    assert not np.shares_memory(a.actor_params, b.actor_params)
    assert not np.shares_memory(a.value_params, b.value_params)
