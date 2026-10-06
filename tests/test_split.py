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


def test_seed_claims_split_sizes_and_no_overlap():
    dataset = load_dataset()
    train, held_out = _deterministic_split(dataset)
    assert len(dataset) == 693
    assert (len(train), len(held_out)) == (554, 139)
    assert {s["claim"] for s in train}.isdisjoint({s["claim"] for s in held_out})


def test_seed_claims_have_binary_labels_and_unique_ids():
    dataset = load_dataset()
    assert {s["label"] for s in dataset} <= {0.0, 1.0}
    assert len({s["id"] for s in dataset}) == len(dataset)


def test_seed_claims_have_one_known_duplicate_claim_text():
    texts = [s["claim"] for s in load_dataset()]
    assert len(texts) - len(set(texts)) == 1


def test_collect_main_seed_controls_randomness_after_split_load(monkeypatch, tmp_path):
    from evid_rl_env.cli import collect_trajectories as collect

    draws = []

    def fake_collect(env, n_episodes, top_k_percent):
        draws.append(random.random())
        return []

    monkeypatch.setattr(collect, "ClaimEnv", lambda dataset: None)
    monkeypatch.setattr(collect, "collect_reward_filtered", fake_collect)
    monkeypatch.setattr(collect, "_write_jsonl", lambda trajectories, path: None)
    monkeypatch.setattr(collect, "_print_summary", lambda trajectories, path: None)

    for seed in ("7", "8"):
        monkeypatch.setattr(
            "sys.argv",
            ["evid-collect", "--mode", "reward_filtered", "--seed", seed,
             "--output", str(tmp_path / "out.jsonl")],
        )
        collect.main()

    assert draws[0] != draws[1]
