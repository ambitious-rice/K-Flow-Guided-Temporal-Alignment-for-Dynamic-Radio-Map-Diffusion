import json
import pytest
from experimental.noise_scratch_full.early_stop import EarlyStop

CONFIG = dict(min_steps=8000, patience=6, mse_relative_delta=.002, alignment_absolute_delta=.002)


def stats(score=1., gain=0.):
    return [dict(correct_mse=score, alignment_gain=gain)]


def test_protected_period_then_six_stalled_validations():
    monitor = EarlyStop(CONFIG)
    for step in [0, 250, 500, *range(1000, 14000, 1000)]:
        assert not monitor.update(step, stats())
    assert monitor.update(14000, stats())


@pytest.mark.parametrize('improved', [stats(.99, 0.), stats(1., .003)])
def test_either_objective_resets_patience_and_resume_preserves_state(improved):
    monitor = EarlyStop(CONFIG)
    monitor.update(8000, stats())
    for step in range(9000, 14000, 1000):
        assert not monitor.update(step, stats())
    restored = EarlyStop(CONFIG, json.loads(json.dumps(monitor.state)))
    assert not restored.update(14000, improved)
    assert restored.state['bad_checks'] == 0
    for step in range(15000, 20000, 1000):
        assert not restored.update(step, improved)
    assert restored.update(20000, improved)


def test_small_changes_do_not_reset_patience_and_duplicate_checks_rejected():
    monitor = EarlyStop(CONFIG)
    monitor.update(8000, stats())
    monitor.update(9000, stats(.999, .001))
    assert monitor.state['bad_checks'] == 1
    with pytest.raises(ValueError):
        monitor.update(9000, stats())
    with pytest.raises(ValueError):
        monitor.update(10000, stats(float('nan')))
