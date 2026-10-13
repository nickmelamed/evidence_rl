import json

import numpy as np
import pytest

from evid_rl_env.agent import policy as policy_module
from evid_rl_env.agent.policy import ActorCriticPolicy
from evid_rl_env.cli import eval_debate

DATASET = [
    {"claim": f"Held-out claim {i}.", "search_query": f"held-out claim {i}", "label": i % 2}
    for i in range(4)
]


class _StubLLM:
    model_name = "org/actor"


def test_load_can_reuse_an_existing_llm_client(tmp_path, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a second client was built")

    original = ActorCriticPolicy(n_actions=13, state_dim=4, llm=_StubLLM())
    original.actor_params[:] = 0.5
    path = str(tmp_path / "p.npz")
    original.save(path)

    monkeypatch.setattr(policy_module, "LLMClient", refuse)
    shared = _StubLLM()
    loaded = ActorCriticPolicy.load(path, llm=shared)
    assert loaded.llm is shared
    np.testing.assert_array_equal(loaded.actor_params, original.actor_params)


def _write_run(tmp_path, turns_per_side=5):
    run = tmp_path / "debate_run"
    run.mkdir()
    (run / "config.json").write_text(json.dumps({"turns_per_side": turns_per_side, "seed": 3}))
    for name in ("policy_support.npz", "policy_contradict.npz"):
        (run / name).write_bytes(b"")
    return run


@pytest.fixture
def patched(monkeypatch, fake_policy, make_env):
    loaded = []

    class _Loader:
        @staticmethod
        def load(path, seed=42, llm=None):
            policy = fake_policy(state_dim=4, seed=len(loaded) + 1)
            policy.llm = llm if llm is not None else object()
            loaded.append((path, llm, policy))
            return policy

    monkeypatch.setattr(eval_debate, "ActorCriticPolicy", _Loader)
    monkeypatch.setattr(eval_debate, "ClaimEnv",
                        lambda dataset, seed: make_env(dataset=dataset, seed=seed))
    return loaded


def test_evaluate_run_writes_results_next_to_the_checkpoints(tmp_path, patched):
    run = _write_run(tmp_path)
    np.random.seed(0)
    result = eval_debate.evaluate_run(str(run), DATASET, n_episodes=3, seed=3)

    written = json.loads((run / "eval_results.json").read_text())
    assert written["n_episodes"] == 3
    assert written["evaluator"] == "debate_pair"
    assert written["baselines"]["rl"]["mean"] == pytest.approx(
        round(result["eval/mean_reward_raw"], 4))
    assert written["baselines"]["rl"]["std"] == pytest.approx(
        round(result["eval/std_reward_raw"], 4))


def test_both_roles_are_loaded_in_order_and_share_one_client(tmp_path, patched):
    run = _write_run(tmp_path)
    np.random.seed(0)
    eval_debate.evaluate_run(str(run), DATASET, n_episodes=1, seed=3)

    (first_path, first_llm, first), (second_path, second_llm, second) = patched
    assert first_path.endswith("policy_support.npz")
    assert second_path.endswith("policy_contradict.npz")
    assert first_llm is None
    assert second_llm is first.llm
    assert second.llm is first.llm


def test_the_turn_count_comes_from_the_run_config(tmp_path, patched):
    run = _write_run(tmp_path, turns_per_side=3)
    np.random.seed(0)
    result = eval_debate.evaluate_run(str(run), DATASET, n_episodes=2, seed=3)
    assert result["eval/mean_steps"] == 6


def test_a_missing_checkpoint_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        eval_debate.evaluate_run(str(tmp_path), DATASET, n_episodes=1, seed=3)
