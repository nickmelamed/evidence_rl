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


def test_the_budget_comes_from_the_run_config(tmp_path, patched):
    run = _write_run(tmp_path)
    cfg = json.loads((run / "config.json").read_text())
    cfg["max_tokens_per_episode"] = 1
    (run / "config.json").write_text(json.dumps(cfg))
    np.random.seed(0)
    result = eval_debate.evaluate_run(str(run), DATASET, n_episodes=2, seed=3)
    assert result["eval/mean_steps"] < 10


def _gold_patches(monkeypatch, make_llm_judge, make_env, patched):
    built = {}

    def fake_build_arbiter(model, seed):
        built["arbiter"] = model
        return make_llm_judge()

    class _Client:
        def __init__(self, model_name, seed):
            built["gold"] = model_name

    monkeypatch.setattr(eval_debate, "build_arbiter", fake_build_arbiter)
    monkeypatch.setattr(eval_debate, "JudgeLLMClient", _Client)
    monkeypatch.setattr(
        eval_debate, "LLMJudge",
        lambda client, cache_path: make_llm_judge(
            '{"LCS": 0.3, "ESS": 0.3, "GRS": 0.7, "COMP": 0.3, "BIAS": 0.7, "confidence": 0.9}'))
    monkeypatch.setattr(
        eval_debate, "ClaimEnv",
        lambda dataset, seed, judge_model=None, llm_judge=None, **kw:
            make_env(dataset=dataset, seed=seed))
    return built


def test_gold_audit_appends_a_row_for_the_run(tmp_path, patched, monkeypatch,
                                              make_llm_judge, make_env):
    run = _write_run(tmp_path)
    cfg = json.loads((run / "config.json").read_text())
    cfg["arbiter_model"] = "org/arbiter"
    (run / "config.json").write_text(json.dumps(cfg))
    built = _gold_patches(monkeypatch, make_llm_judge, make_env, patched)

    np.random.seed(0)
    summary = eval_debate.gold_audit_run(str(run), DATASET, "org/gold", n_episodes=2, seed=3)

    rows = [json.loads(line) for line in (run / "gold_eval.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["n_episodes"] == 2
    assert rows[0]["checkpoint"] == str(run)
    assert rows[0]["n_scored"] == summary["n_scored"]
    assert built == {"arbiter": "org/arbiter", "gold": "org/gold"}


def test_gold_audit_refuses_a_gold_judge_that_is_the_arbiter(tmp_path, patched):
    run = _write_run(tmp_path)
    cfg = json.loads((run / "config.json").read_text())
    cfg["arbiter_model"] = "org/same"
    (run / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="gold judge"):
        eval_debate.gold_audit_run(str(run), DATASET, "org/same", n_episodes=1, seed=3)


def test_gold_audit_needs_the_arbiter_recorded_in_the_run_config(tmp_path, patched):
    run = _write_run(tmp_path)
    with pytest.raises(ValueError, match="arbiter_model"):
        eval_debate.gold_audit_run(str(run), DATASET, "org/gold", n_episodes=1, seed=3)
