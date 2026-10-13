import json
import random

import pytest

from evid_rl_env.cli import collect_trajectories as collect
from evid_rl_env.environment.actions import ACTIONS, Actions

TRAIN = [{"claim": "Train claim.", "search_query": "train"}]


def _record(claim="Train claim.", mode="llm_annotator", actions=("SELECT", "FINALIZE")):
    return {
        "claim": claim,
        "annotator_model": "some-model",
        "mode": mode,
        "collected_at": "2026-01-01T00:00:00+00:00",
        "total_reward": 9.9,
        "steps": [{"obs": "o", "action": ACTIONS.index(getattr(Actions, a)), "reward": 1.0}
                  for a in actions],
    }


class _FakeAnnotator:
    def __init__(self):
        self.calls = 0

    def select_action(self, obs, descriptions):
        self.calls += 1
        return 3


@pytest.fixture
def factory(make_env):
    return lambda dataset: make_env(dataset=dataset)


def test_rescore_replays_actions_and_records_provenance(factory):
    out = collect.rescore_trajectories([_record()], TRAIN, factory)
    assert len(out) == 1
    rec = out[0]
    assert rec["mode"] == "rescored"
    assert rec["source_mode"] == "llm_annotator"
    assert rec["annotator_model"] == "some-model"
    assert rec["source_total_reward"] == 9.9
    assert [s["action"] for s in rec["steps"]] == [
        ACTIONS.index(Actions.SELECT), ACTIONS.index(Actions.FINALIZE)
    ]
    assert rec["total_reward"] == pytest.approx(sum(s["reward"] for s in rec["steps"]))
    assert rec["steps"][0]["source_reward"] == 1.0
    assert rec["steps_dropped"] == 0
    assert "relabel_model" not in rec


def test_rescore_skips_claims_outside_the_training_split(factory):
    records = [_record(claim="Held-out claim."), _record()]
    out = collect.rescore_trajectories(records, TRAIN, factory)
    assert [r["claim"] for r in out] == ["Train claim."]


def test_rescore_skips_unrecognized_or_missing_mode(factory):
    records = [_record(mode="eval"), {**_record(), "mode": None}]
    assert collect.rescore_trajectories(records, TRAIN, factory) == []


def test_rescore_accepts_its_own_output_as_input(factory):
    first = collect.rescore_trajectories([_record()], TRAIN, factory)
    second = collect.rescore_trajectories(first, TRAIN, factory)
    assert second[0]["source_mode"] == "rescored"


def test_rescore_drops_steps_after_the_episode_ends(factory):
    # FINALIZE ends the episode only after step 2, earlier ones are penalised and continue
    rec = _record(actions=("SELECT", "SELECT", "FINALIZE", "SELECT", "SELECT"))
    out = collect.rescore_trajectories([rec], TRAIN, factory)[0]
    assert len(out["steps"]) == 3
    assert out["steps_dropped"] == 2


def test_rescore_is_repeatable_and_restores_global_rng(factory):
    records = [_record(actions=("SELECT", "SELECT", "FINALIZE"))]
    random.seed(123)
    before = random.getstate()
    a = collect.rescore_trajectories(records, TRAIN, factory, seed=5)
    assert random.getstate() == before
    b = collect.rescore_trajectories(records, TRAIN, factory, seed=5)
    assert [s["reward"] for s in a[0]["steps"]] == [s["reward"] for s in b[0]["steps"]]


def test_relabel_adds_annotator_action_without_changing_the_executed_one(factory):
    ann = _FakeAnnotator()
    out = collect.rescore_trajectories(
        [_record()], TRAIN, factory, annotator=ann, annotator_model="strong-model"
    )[0]
    assert ann.calls == 2
    assert all(s["annotator_action"] == 3 for s in out["steps"])
    assert out["steps"][0]["action"] == ACTIONS.index(Actions.SELECT)
    assert out["relabel_model"] == "strong-model"


def test_main_rescore_requires_input_and_refuses_to_overwrite(monkeypatch, tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_text(json.dumps(_record()) + "\n")
    monkeypatch.setattr("sys.argv", ["evid-collect", "--mode", "rescore"])
    with pytest.raises(SystemExit):
        collect.main()
    monkeypatch.setattr(
        "sys.argv",
        ["evid-collect", "--mode", "rescore", "--input", str(path), "--output", str(path)],
    )
    with pytest.raises(SystemExit):
        collect.main()
    assert json.loads(path.read_text())["mode"] == "llm_annotator"


def test_best_rollouts_accepts_rescored_files(tmp_path):
    rec = {**_record(), "mode": "rescored", "total_reward": 1.0}
    (tmp_path / "r.jsonl").write_text(json.dumps(rec) + "\n")
    out = collect.collect_best_rollouts(0.5, logs_dir=str(tmp_path))
    assert len(out) == 1
