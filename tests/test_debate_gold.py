import json
import os
from pathlib import Path

import numpy as np
import pytest

from evid_rl_env.agent.config_loader import load_config
from evid_rl_env.agent.debate_gold import build_debate_gold_evaluator
from evid_rl_env.agent.debate_trainer import DebateTrainer
from evid_rl_env.environment.actions import ACTIONS, Actions

DATASET = [
    {"claim": f"Gold audit claim {i}.", "search_query": f"gold audit claim {i}", "label": i % 2}
    for i in range(4)
]
_CONFIGS_DIR = Path(__file__).parent.parent / "configs"
ARGUMENT = {"argument": "an argument", "evidence_ids": [0], "tokens": 3}


class _ScriptedPolicy:
    """Selects evidence, assigns a confidence, then argues, so every episode reaches FINALIZE."""

    def __init__(self, side):
        self.side = side
        self.actions = ACTIONS
        self.n_actions = len(ACTIONS)
        self.last_entropy = 0.0
        self.last_probs = np.ones(self.n_actions) / self.n_actions
        self.last_features = np.zeros(4)
        self.last_mask = np.ones(self.n_actions, dtype=bool)
        self.llm = None

    def reset_episode_cache(self):
        pass

    def get_value(self, state):
        return 0.0

    def act(self, state, greedy=False, allowed_actions=None, **kwargs):
        if not state.selected_evidence:
            return Actions.SELECT, 0, ACTIONS.index(Actions.SELECT)
        if state.confidence is None and allowed_actions == [Actions.ASSIGN_CONFIDENCE]:
            return Actions.ASSIGN_CONFIDENCE, 0.8, ACTIONS.index(Actions.ASSIGN_CONFIDENCE)
        action = Actions.SUPPORT if self.side == "support" else Actions.CONTRADICT
        return action, ARGUMENT, ACTIONS.index(action)

    def save(self, path):
        np.savez_compressed(path, actor_params=np.zeros((4, len(ACTIONS))),
                            value_params=np.zeros(4))


@pytest.fixture(autouse=True)
def _sandbox_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _gold_judge(make_llm_judge, prompts):
    judge = make_llm_judge(
        '{"LCS": 0.3, "ESS": 0.3, "GRS": 0.7, "COMP": 0.3, "BIAS": 0.7, "confidence": 0.9}'
    )
    original = judge.llm.generate_structured
    judge.llm.generate_structured = lambda prompt: (prompts.append(prompt), original(prompt))[1]
    return judge


def test_the_gold_judge_scores_the_two_sided_transcript(make_env, make_llm_judge):
    prompts = []
    evaluator = build_debate_gold_evaluator(
        make_env(dataset=DATASET), _ScriptedPolicy("support"), _ScriptedPolicy("contradict"),
        [_gold_judge(make_llm_judge, prompts)], n_episodes=2, turns_per_side=5, max_tokens=None,
    )
    np.random.seed(0)
    result = evaluator.evaluate()

    assert result["n_episodes"] == 2
    assert result["n_scored"] == 2
    assert prompts
    assert "SUPPORT: an argument" in prompts[0]
    assert "CONTRADICT: an argument" in prompts[0]


def test_the_proxy_side_is_the_arbiters_final_score(make_env, make_llm_judge):
    env = make_env(dataset=DATASET)
    evaluator = build_debate_gold_evaluator(
        env, _ScriptedPolicy("support"), _ScriptedPolicy("contradict"),
        [_gold_judge(make_llm_judge, [])], n_episodes=1, turns_per_side=5, max_tokens=None,
    )
    np.random.seed(0)
    result = evaluator.evaluate()
    # the arbiter mock returns fixed good scores and the gold mock fixed poor ones
    assert result["proxy_reward_mean"] > result["gold_reward_mean"]


def test_the_trainer_writes_a_gold_row_on_its_cadence(make_env, make_llm_judge):
    _, config = load_config(str(_CONFIGS_DIR / "ppo_baseline.yaml"))
    config.gold_eval_every = 2
    config.gold_judge_model = "org/gold"
    config.gold_eval_n_episodes = 2
    trainer = DebateTrainer(
        env=make_env(dataset=DATASET), support_policy=_ScriptedPolicy("support"),
        contradict_policy=_ScriptedPolicy("contradict"), config=config, episodes=4,
        exp_name="gold_cadence", seed=0, eval_dataset=DATASET, eval_every=1,
        gold_judges=[_gold_judge(make_llm_judge, [])],
    )
    trainer.rl = {role: type("NoUpdate", (), {"update": lambda self, t: None})()
                  for role in ("support", "contradict")}
    trainer.train()

    path = os.path.join(trainer.tracker.base_dir, "gold_eval.jsonl")
    rows = [json.loads(line) for line in open(path)]
    assert [r["episode"] for r in rows] == [2, 4]
    assert all(r["n_scored"] > 0 for r in rows)


def test_a_gold_round_does_not_disturb_the_training_rng(make_env, make_llm_judge):
    _, config = load_config(str(_CONFIGS_DIR / "ppo_baseline.yaml"))
    config.gold_judge_model = "org/gold"
    config.gold_eval_n_episodes = 2
    trainer = DebateTrainer(
        env=make_env(dataset=DATASET), support_policy=_ScriptedPolicy("support"),
        contradict_policy=_ScriptedPolicy("contradict"), config=config, episodes=1,
        exp_name="gold_rng", seed=0, eval_dataset=DATASET, eval_every=1,
        gold_judges=[_gold_judge(make_llm_judge, [])],
    )
    before = np.random.get_state()[1].copy()
    trainer._run_gold_round(0)
    np.testing.assert_array_equal(before, np.random.get_state()[1])


def test_the_trainer_refuses_a_gold_judge_that_is_the_arbiter(make_env):
    _, config = load_config(str(_CONFIGS_DIR / "ppo_baseline.yaml"))
    config.gold_judge_model = "org/same"
    with pytest.raises(ValueError, match="gold judge"):
        DebateTrainer(
            env=make_env(dataset=DATASET), support_policy=_ScriptedPolicy("support"),
            contradict_policy=_ScriptedPolicy("contradict"), config=config, episodes=1,
            exp_name="gold_same", seed=0, eval_dataset=DATASET, arbiter_model="org/same",
        )
