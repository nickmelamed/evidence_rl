import random

from evid_rl_env.cli.collect_trajectories import _load_train_split
from evid_rl_env.cli.eval import _deterministic_split
from evid_rl_env.data.dataset import load_dataset


def _key(sample):
    return sample.get("id", sample["claim"])


def test_deterministic_split_is_80_20_and_disjoint():
    dataset = [{"id": i, "claim": f"c{i}"} for i in range(50)]
    train, held_out = _deterministic_split(dataset)
    assert len(train) == 40
    assert len(held_out) == 10
    assert {_key(s) for s in train}.isdisjoint({_key(s) for s in held_out})
    assert {_key(s) for s in train + held_out} == {_key(s) for s in dataset}


def test_deterministic_split_is_stable_across_calls():
    dataset = [{"id": i, "claim": f"c{i}"} for i in range(50)]
    assert _deterministic_split(dataset) == _deterministic_split(dataset)


def test_deterministic_split_restores_global_rng_state():
    random.seed(123)
    before = random.getstate()
    _deterministic_split([{"id": i, "claim": f"c{i}"} for i in range(20)])
    assert random.getstate() == before


def test_collect_train_split_matches_eval_train_split():
    train, _ = _deterministic_split(load_dataset())
    assert _load_train_split(None) == train


def test_collect_train_split_restores_global_rng_state():
    random.seed(7)
    before = random.getstate()
    _load_train_split(None)
    assert random.getstate() == before
