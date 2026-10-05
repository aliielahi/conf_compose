import pytest

from matu.fit_to_uncertainty import uncertainty_from_legacy_fit, uncertainty_from_matu_result


def test_uncertainty_from_legacy_fit():
    assert uncertainty_from_legacy_fit([0.9, 0.8]) == pytest.approx(0.3)


def test_uncertainty_from_matu_result_prefers_relative_loss():
    value = {"relative_loss": [0.1, 0.2], "fit": [0.99, 0.99]}
    assert uncertainty_from_matu_result(value) == pytest.approx(0.3)
