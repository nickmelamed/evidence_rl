import pytest

from evid_rl_env.agent.config import MultiAgentPPOConfig
from evid_rl_env.cli import train as train_cli


class _Policy:
    def __init__(self, n_actions, model_name=None, seed=None, llm=None):
        self.llm = llm if llm is not None else object()
        self.model_name = model_name


def test_train_multiagent_wires_the_arbiter_and_one_shared_actor(monkeypatch):
    built = {}
    monkeypatch.setattr(train_cli, "build_arbiter",
                        lambda model, seed: built.setdefault("arbiter", (model, seed)) and "ARBITER")

    class _Env:
        def __init__(self, dataset, judge_model, seed, llm_judge):
            built["env"] = (judge_model, seed, llm_judge)

    class _Trainer:
        def __init__(self, **kwargs):
            built["trainer"] = kwargs

        def train(self):
            built["trained"] = True

    monkeypatch.setattr(train_cli, "ClaimEnv", _Env)
    monkeypatch.setattr(train_cli, "ActorCriticPolicy", _Policy)
    monkeypatch.setattr(train_cli, "DebateTrainer", _Trainer)

    config = MultiAgentPPOConfig()
    config.arbiter_model = "org/arbiter"
    config.judge_model = "org/labeler"
    config.actor_model = "org/actor"
    config.turns_per_side = 4
    train_cli.train_multiagent(["claim"], config, episodes=3, seed=9, exp_name="x", curriculum=None)

    assert built["arbiter"] == ("org/arbiter", 9)
    assert built["env"] == ("org/labeler", 9, "ARBITER")
    kwargs = built["trainer"]
    assert kwargs["support_policy"].llm is kwargs["contradict_policy"].llm
    assert kwargs["support_policy"] is not kwargs["contradict_policy"]
    assert kwargs["turns_per_side"] == 4
    assert kwargs["arbiter_model"] == "org/arbiter"
    assert kwargs["episodes"] == 3
    assert built["trained"] is True


def test_the_cli_routes_the_multiagent_algo_to_train_multiagent(monkeypatch):
    called = {}
    monkeypatch.setattr(train_cli, "load_dataset", lambda: [{"claim": "c"}] * 10)
    monkeypatch.setattr(train_cli, "split_dataset", lambda d: (d[:8], d[8:]))
    monkeypatch.setattr(train_cli, "warm_cache", lambda d: None)
    monkeypatch.setattr(train_cli, "train_multiagent",
                        lambda *args: called.setdefault("args", args))
    train_cli.train(2, config_path="configs/ppo_multiagent_debate.yaml", seed=5,
                    exp_name="routed")
    train_data, config, episodes, seed, exp_name, _ = called["args"]
    assert len(train_data) == 8
    assert config.arbiter_model == "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    assert (episodes, seed, exp_name) == (2, 5, "routed")


def test_train_multiagent_refuses_an_arbiter_that_is_also_the_gold_judge(monkeypatch):
    monkeypatch.setattr(train_cli, "build_arbiter",
                        lambda model, seed: pytest.fail("the arbiter must not be built"))
    config = MultiAgentPPOConfig()
    config.arbiter_model = "org/shared"
    config.gold_judge_model = "org/shared"
    with pytest.raises(ValueError, match="gold judge"):
        train_cli.train_multiagent(["claim"], config, episodes=1, seed=0, exp_name="x",
                                   curriculum=None)
