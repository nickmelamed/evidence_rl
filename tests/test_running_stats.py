import numpy as np
import pytest

from evid_rl_env.utils.running_stats import RunningMeanStd


def test_running_mean_and_var_approach_true_values():
    rms = RunningMeanStd()
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 200, dtype=np.float64)
    rms.update(data)
    assert rms.mean == pytest.approx(np.mean(data), abs=1e-3)
    assert rms.var == pytest.approx(np.var(data), abs=1e-2)


def test_update_raises_while_locked():
    rms = RunningMeanStd()
    rms._locked = True
    with pytest.raises(AssertionError):
        rms.update([1.0, 2.0, 3.0])


def test_unlocked_update_still_works():
    rms = RunningMeanStd()
    rms._locked = True
    rms._locked = False
    rms.update([1.0, 2.0, 3.0])  # should not raise


def test_normalize_centers_the_mean_near_zero():
    rms = RunningMeanStd()
    data = np.random.RandomState(0).normal(loc=5.0, scale=2.0, size=2000)
    rms.update(data)
    assert rms.normalize(rms.mean) == pytest.approx(0.0, abs=1e-6)


def test_incremental_updates_match_one_batch_update():
    rms_batch = RunningMeanStd()
    rms_incremental = RunningMeanStd()
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    rms_batch.update(data)
    for x in data:
        rms_incremental.update([x])

    assert rms_incremental.mean == pytest.approx(rms_batch.mean, abs=1e-6)
    assert rms_incremental.var == pytest.approx(rms_batch.var, abs=1e-6)
