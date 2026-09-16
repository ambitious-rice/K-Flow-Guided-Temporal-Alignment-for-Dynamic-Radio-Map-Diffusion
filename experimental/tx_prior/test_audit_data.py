import numpy as np

from experimental.tx_prior.audit_data import array_summary


def test_array_summary_preserves_range_and_energy():
    result = array_summary(np.array([[0, 1], [0.5, 0.5]], dtype=np.float32))
    assert result["shape"] == [2, 2]
    assert result["finite"]
    assert result["min"] == 0 and result["max"] == 1
    assert result["mean"] == 0.5 and result["energy"] == 0.375
    assert result["zero_fraction"] == 0.25 and result["one_fraction"] == 0.25


def test_array_summary_detects_nonfinite_data():
    result = array_summary(np.array([0, np.nan], dtype=np.float32))
    assert not result["finite"]
