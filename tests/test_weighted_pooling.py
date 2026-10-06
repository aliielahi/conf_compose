import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import expit

from conf_compose.composition.pooling import FittedBLP, fit_blp, fit_kahn, fit_logistic_pool
from conf_compose.constants import COMPOSITION_PANELS


def informative_scores():
    values = np.linspace(-2.5, 2.5, 40)
    labels = np.tile([0, 0, 0, 1, 1], 8)
    good = expit(np.where(labels, 1.2, -1.2) + values * 0.35)
    noise = expit(np.sin(np.arange(40)))
    return np.column_stack([good, noise]), labels


def test_blp_identity_and_monotonicity():
    fit = FittedBLP("blp", 2, None, "fitted", 10, 3, weights=(0.25, 0.75))
    assert fit.predict([0.2, 0.8]).score == pytest.approx(0.65)
    assert fit.predict([0.5, 0.8]).score > fit.predict([0.2, 0.8]).score
    assert fit.predict([0, 0]).logit < fit.predict([1, 1]).logit


def test_blp_unequal_weights_and_equal_control():
    scores, labels = informative_scores()
    fit = fit_blp(scores, labels)
    equal = fit_blp(scores, labels, equal_weights=True)
    assert fit.status == equal.status == "fitted"
    assert sum(fit.weights) == pytest.approx(1)
    assert min(fit.weights) >= 0
    assert fit.weights[0] > fit.weights[1]
    assert equal.weights == pytest.approx([0.5, 0.5])
    assert fit.parameters["fit_nll"] < equal.parameters["fit_nll"]
    for value in (fit, equal):
        json.dumps(value.parameters, allow_nan=False)
        assert np.isfinite(value.predict([0, 1]).logit)


def test_logistic_unequal_signed_weights_and_regularization():
    values = np.array([-2, -1, -0.5, 0.5, 1, 2])
    scores = expit(np.column_stack([values, -values]))
    labels = [0, 0, 0, 1, 1, 1]
    fit = fit_logistic_pool(scores, labels)
    strong = fit_logistic_pool(scores, labels, l2=10)
    assert fit.status == strong.status == "fitted"
    assert fit.weights[0] > 0 > fit.weights[1]
    assert np.linalg.norm(strong.weights) < np.linalg.norm(fit.weights)
    assert fit.predict([0.8, 0.2]).score > fit.predict([0.2, 0.8]).score
    json.dumps(fit.parameters, allow_nan=False)


def test_logistic_constant_streams_and_duplicate_streams():
    fit = fit_logistic_pool([[0.8, 0.8]] * 10, [0] * 2 + [1] * 8)
    assert fit.predict([0.8, 0.8]).score == pytest.approx(0.8)
    assert fit.weights == pytest.approx([0, 0])
    scores = expit(np.column_stack([np.linspace(-2, 2, 20)] * 5))
    duplicate = fit_logistic_pool(scores, [0] * 10 + [1] * 10)
    assert np.isfinite(duplicate.weights).all()
    assert duplicate.weights == pytest.approx([duplicate.weights[0]] * 5)


@pytest.mark.parametrize("fitter", [fit_blp, fit_logistic_pool])
def test_column_permutation_and_degenerate_fits(fitter):
    scores, labels = informative_scores()
    fit = fitter(scores, labels)
    reverse = fitter(scores[:, ::-1], labels)
    assert reverse.predict([0.4, 0.7]).score == pytest.approx(fit.predict([0.7, 0.4]).score, abs=1e-5)
    for matrix, outcomes in [(np.empty((0, 2)), []), ([[0.6, 0.5]] * 4, [1] * 4)]:
        degenerate = fitter(matrix, outcomes)
        assert degenerate.status == "insufficient_classes"
        assert degenerate.predict([0.7, 0.7]).score is None
    with pytest.raises(ValueError):
        fitter([[float("nan"), 0.5]], [1])
    with pytest.raises(ValueError):
        fit.predict([0.2])
    with pytest.raises(ValueError):
        fit.predict([0.2, 1.1])


def test_diagonal_kahn_only_removes_off_diagonal_covariance():
    scores, labels = informative_scores()
    full = fit_kahn(scores, labels, shrinkage=0.2)
    diagonal = fit_kahn(scores, labels, shrinkage=0.2, covariance_mode="diagonal")
    covariance = np.array(full.parameters["regularized_covariance"])
    assert np.allclose(diagonal.parameters["regularized_covariance"], np.diag(np.diag(covariance)))
    assert diagonal.parameters["prior"] == full.parameters["prior"]
    assert diagonal.parameters["shrinkage"] == full.parameters["shrinkage"]


@pytest.mark.parametrize("module", ["blp", "logistic"])
def test_failed_optimizers_are_not_silently_used(monkeypatch, module):
    import importlib
    target = importlib.import_module(f"conf_compose.composition.pooling.{module}")
    def failure(function, initial, **kwargs):
        return SimpleNamespace(success=False, message="failed", x=initial, fun=1.0, nit=0)
    monkeypatch.setattr(target, "minimize", failure)
    scores, labels = informative_scores()
    fit = (target.fit_blp if module == "blp" else target.fit_logistic_pool)(scores, labels)
    assert fit.status == "optimization_failed"
    assert fit.predict([0.6, 0.7]).score is None


def test_judge_panels_and_sweep_dry_run():
    assert len(COMPOSITION_PANELS) == 15
    assert [len(panel) for panel in COMPOSITION_PANELS] == [2] * 4 + [3] * 4 + [4] * 3 + [5] * 2 + [6] * 2
    assert all(len(panel) == len(set(panel)) for panel in COMPOSITION_PANELS)
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "runs/experiment02-voting_composition/sweep_learned.py", "--dry-run"],
                            cwd=root, capture_output=True, text=True, check=True)
    assert "150 atomic rows" in result.stdout
    assert "estimators=['cons', 'seq']" in result.stdout


def test_beta_ranking_retains_close_values_and_equal_pool_ties():
    fit = FittedBLP("blp_equal", 3, None, "fitted", 100, 20,
                    parameters={"equal_weights": True}, weights=(1 / 3,) * 3, alpha=80, beta=0.1)
    inputs = [[0.999, 0.999, 0.999], [0.9999, 0.9999, 0.9999], [0.2, 0.4, 0.6], [0.6, 0.4, 0.2]]
    for scores in inputs:
        assert fit.predict(scores).ranking_score == sum(scores) / len(scores)
    assert fit.predict(inputs[1]).ranking_score > fit.predict(inputs[0]).ranking_score
