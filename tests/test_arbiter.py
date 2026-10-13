import pytest

from evid_rl_env.environment.state import Evidence
from evid_rl_env.judge.arbiter import Arbiter, build_arbiter

EVIDENCE = [Evidence(id=0, text="A trial.", label="neutral")]


def test_arbiter_scores_like_the_judge_it_wraps(make_llm_judge):
    judge = make_llm_judge()
    arbiter = Arbiter(judge, model_name="stub-arbiter")
    expected = judge.compute_reward("claim", "SUPPORT: a CONTRADICT: b", EVIDENCE)
    assert arbiter.compute_reward("claim", "SUPPORT: a CONTRADICT: b", EVIDENCE) == expected


def test_arbiter_gives_zero_reward_to_an_empty_transcript(make_llm_judge):
    arbiter = Arbiter(make_llm_judge(), model_name="stub-arbiter")
    reward, _ = arbiter.compute_reward("claim", "  ", EVIDENCE)
    assert reward == 0.0


def test_arbiter_prompt_contains_both_sides_of_the_transcript(make_llm_judge):
    judge = make_llm_judge()
    seen = []
    original = judge.llm.generate_structured

    def recording(prompt):
        seen.append(prompt)
        return original(prompt)

    judge.llm.generate_structured = recording
    Arbiter(judge, model_name="stub-arbiter").compute_reward(
        "claim", "SUPPORT: for it CONTRADICT: against it", EVIDENCE
    )
    assert "SUPPORT: for it" in seen[0]
    assert "CONTRADICT: against it" in seen[0]


def test_build_arbiter_uses_its_own_model_seed_and_cache(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    built = {}

    class _Client:
        def __init__(self, model_name, seed):
            built["model_name"] = model_name
            built["seed"] = seed

    monkeypatch.setattr("evid_rl_env.agent.llm_client.JudgeLLMClient", _Client)
    arbiter = build_arbiter("org/arbiter-model", seed=7)

    assert built == {"model_name": "org/arbiter-model", "seed": 7}
    assert arbiter.model_name == "org/arbiter-model"
    assert (tmp_path / "artifacts/cache/judge_cache_org__arbiter-model.sqlite3").exists()


def test_build_arbiter_rejects_an_empty_model_name():
    with pytest.raises(ValueError):
        build_arbiter("", seed=0)
