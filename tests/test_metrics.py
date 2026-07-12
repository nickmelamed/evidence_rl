from evid_rl_env.environment.state import Evidence
from evid_rl_env.judge.metrics import (
    compute_adversarial_contamination,
    compute_contradiction_acknowledgment,
    compute_f1,
    compute_precision,
    compute_recall,
)

SUPPORT = Evidence(id=0, text="support", label="support")
CONTRADICT = Evidence(id=1, text="contradict", label="contradict")
NEUTRAL = Evidence(id=2, text="neutral", label="neutral")
ADVERSARIAL = Evidence(id=3, text="adversarial", label="adversarial")


def test_precision_empty_selection():
    assert compute_precision([]) == 0.0


def test_precision_all_support():
    assert compute_precision([SUPPORT, SUPPORT]) == 1.0


def test_precision_mixed():
    assert compute_precision([SUPPORT, NEUTRAL]) == 0.5


def test_precision_counts_contradict_as_relevant():
    """Contradiction-acknowledgment shouldn't be taxed by precision: selecting
    a contradicting document is exactly what compute_contradiction_acknowledgment
    rewards, so it must not also lower precision."""
    assert compute_precision([SUPPORT, CONTRADICT]) == 1.0


def test_recall_no_support_available_is_vacuously_perfect():
    assert compute_recall([], [NEUTRAL, CONTRADICT]) == 1.0


def test_recall_partial():
    available = [SUPPORT, Evidence(id=4, text="s2", label="support")]
    assert compute_recall([SUPPORT], available) == 0.5


def test_f1_zero_when_nothing_selected():
    assert compute_f1([], [SUPPORT]) == 0.0


def test_f1_perfect_when_all_support_selected_and_available():
    assert compute_f1([SUPPORT], [SUPPORT]) == 1.0


def test_contradiction_acknowledgment_vacuous_when_none_available():
    assert compute_contradiction_acknowledgment([], [SUPPORT, NEUTRAL]) == 1.0


def test_contradiction_acknowledgment_partial():
    available = [CONTRADICT, Evidence(id=5, text="c2", label="contradict")]
    assert compute_contradiction_acknowledgment([CONTRADICT], available) == 0.5


def test_adversarial_contamination_empty_selection():
    assert compute_adversarial_contamination([]) == 0.0


def test_adversarial_contamination_none_selected():
    assert compute_adversarial_contamination([SUPPORT, NEUTRAL]) == 0.0


def test_adversarial_contamination_partial():
    assert compute_adversarial_contamination([ADVERSARIAL, SUPPORT]) == 0.5


def test_adversarial_contamination_all():
    assert compute_adversarial_contamination([ADVERSARIAL]) == 1.0
