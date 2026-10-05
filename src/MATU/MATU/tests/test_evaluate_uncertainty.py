import math

import numpy as np
import pytest

from matu.evaluate_uncertainty import accuracy_from_labels, auarc, auroc_binary, score_from_value


def test_score_from_legacy_fit():
    assert score_from_value([0.9, 0.8, 0.7], "legacy_fit") == pytest.approx(0.6)


def test_accuracy_from_labels():
    assert accuracy_from_labels(["correct", "incorrect", "true", "0"]) == 0.5


def test_metrics_are_finite_on_toy_data():
    accuracy = np.asarray([1.0, 0.5, 0.0])
    uncertainty = np.asarray([0.1, 0.5, 0.9])
    y_error = (accuracy < 1.0).astype(int)

    assert 0.0 <= auroc_binary(y_error, uncertainty) <= 1.0
    assert math.isfinite(auarc(accuracy, uncertainty))
