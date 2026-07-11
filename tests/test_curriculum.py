import pytest

from evid_rl_env.environment.curriculum import Curriculum

DATASET = [
    {"id": "c_a", "claim": "claim a"},
    {"id": "c_b", "claim": "claim b"},
    {"id": "c_c", "claim": "claim c"},
]


def test_unseen_claims_get_max_learning_progress():
    curriculum = Curriculum()
    assert curriculum.learning_progress("c_a") == 1.0


def test_unseen_claims_are_maximally_stale():
    curriculum = Curriculum()
    assert curriculum.staleness("c_a") == 1.0


def test_staleness_is_near_zero_right_after_recording():
    # record() ticks the shared episode counter as part of the same call, so
    # staleness immediately after being recorded is 1/long_window, not
    # exactly 0 — it climbs from there as other claims get recorded.
    curriculum = Curriculum(long_window=10)
    curriculum.record("c_a", 0.5)
    assert curriculum.staleness("c_a") == pytest.approx(0.1)


def test_staleness_grows_as_other_claims_are_recorded():
    curriculum = Curriculum(long_window=10)
    curriculum.record("c_a", 0.5)
    for _ in range(5):
        curriculum.record("c_b", 0.5)
    assert curriculum.staleness("c_a") == pytest.approx(0.6)


def test_stable_performance_drives_learning_progress_to_zero():
    curriculum = Curriculum(short_window=3, long_window=6)
    for _ in range(6):
        curriculum.update("c_a", 0.5)
    assert curriculum.learning_progress("c_a") == 0.0


def test_recent_shift_in_performance_raises_learning_progress():
    curriculum = Curriculum(short_window=3, long_window=6)
    for _ in range(6):
        curriculum.update("c_a", 0.2)
    for _ in range(3):
        curriculum.update("c_a", 0.9)
    assert curriculum.learning_progress("c_a") > 0.0


def test_claim_with_higher_learning_progress_scores_higher():
    curriculum = Curriculum(short_window=3, long_window=6, min_weight=0.0, staleness_coef=0.0)
    for _ in range(6):
        curriculum.update("c_a", 0.5)  # flat -> zero learning progress
        curriculum.update("c_b", 0.2)
    for _ in range(3):
        curriculum.update("c_b", 0.9)  # recent shift -> positive learning progress
    assert curriculum.score("c_b") > curriculum.score("c_a")


def test_sample_returns_an_item_from_the_dataset():
    curriculum = Curriculum()
    for _ in range(20):
        sample = curriculum.sample(DATASET)
        assert sample in DATASET


def test_sample_falls_back_to_claim_text_when_id_is_absent():
    curriculum = Curriculum()
    dataset = [{"claim": "no id here"}]
    sample = curriculum.sample(dataset)
    assert sample == dataset[0]

    curriculum.record("no id here", 0.5)  # keyed by claim text, same fallback record() would use
    assert curriculum.staleness("no id here") == pytest.approx(1 / curriculum.long_window)


def test_update_is_backwards_compatible_alias_for_record():
    curriculum = Curriculum(short_window=3, long_window=6)
    curriculum.update("c_b", 0.7)
    assert list(curriculum._short["c_b"]) == [0.7]
    assert list(curriculum._long["c_b"]) == [0.7]


def test_mean_score_defaults_to_min_weight_before_any_sample_call():
    curriculum = Curriculum(min_weight=0.1)
    assert curriculum.mean_score == pytest.approx(0.1)


def test_mean_score_is_average_score_over_the_last_sampled_dataset():
    curriculum = Curriculum(short_window=3, long_window=6, staleness_coef=0.0)
    curriculum.sample(DATASET)  # populates _known_ids from the full dataset
    expected = sum(curriculum.score(d["id"]) for d in DATASET) / len(DATASET)
    assert curriculum.mean_score == pytest.approx(expected)


def test_mean_score_falls_as_performance_stabilizes_across_all_claims():
    curriculum = Curriculum(short_window=3, long_window=6, min_weight=0.0, staleness_coef=0.0)
    curriculum.sample(DATASET)
    high_progress_score = curriculum.mean_score  # all claims cold-start at learning_progress=1.0

    for claim in DATASET:
        for _ in range(6):
            curriculum.update(claim["id"], 0.5)  # flat performance -> learning progress converges to 0
    curriculum.sample(DATASET)
    stable_score = curriculum.mean_score

    assert stable_score < high_progress_score
