import json
import math

import numpy as np
import pytest

from conf_compose.composition.pooling import fit_kahn, pool_methods
from conf_compose.composition.pooling.kahn import _oas_shrinkage


def probabilities(logits):
    return 1 / (1 + np.exp(-np.asarray(logits, dtype=float)))


def test_kahn_agrees_with_gaussian_likelihood_ratio():
    residuals = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=float)
    transform = np.array([[1, 0], [0.5, math.sqrt(0.75)]])
    residuals = residuals @ transform.T
    mu0, mu1 = np.array([-1, 0.5]), np.array([1, 0.5])
    scores = probabilities(np.vstack([residuals + mu0, residuals + mu1]))
    fit = fit_kahn(scores, [0] * 4 + [1] * 4, shrinkage=0, ridge=1e-10)
    covariance = (8 / 6) * (transform @ transform.T)
    assert np.allclose(fit.parameters["covariance"], covariance)
    assert np.allclose(fit.weights, [2, -1], atol=1e-8)
    assert fit.intercept == pytest.approx(0.5, abs=1e-8)
    for point in ([0, 0.5], [1, 0.5], [-1, 2]):
        point = np.asarray(point)
        log_density0 = -0.5 * (point - mu0) @ np.linalg.solve(covariance, point - mu0)
        log_density1 = -0.5 * (point - mu1) @ np.linalg.solve(covariance, point - mu1)
        expected = float(probabilities(log_density1 - log_density0))
        assert fit.predict(probabilities(point)).score == pytest.approx(expected, abs=1e-8)


def test_kahn_conditions_covariance_on_correctness():
    residuals = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    logits = np.vstack([residuals - 3, residuals + 3])
    fit = fit_kahn(probabilities(logits), [0] * 4 + [1] * 4)
    assert np.corrcoef(logits.T)[0, 1] > 0.8
    assert fit.parameters["covariance"][0][1] == pytest.approx(0, abs=1e-12)


def test_oas_finite_dimension_formula_and_isotropic_cases():
    covariance = np.array([[2, 1], [1, 2]], dtype=float)
    assert _oas_shrinkage(covariance, 10) == pytest.approx(0.8)
    assert _oas_shrinkage(np.eye(3), 10) == 1
    assert _oas_shrinkage(np.zeros((3, 3)), 10) == 1
    assert _oas_shrinkage(np.array([[2.0]]), 10) == 1


def test_kahn_can_reverse_a_misleading_stream_and_change_ranking():
    logits = np.array([[-2, 2], [-1, 1], [-0.5, 0.5], [0.5, -0.5], [1, -1], [2, -2]])
    fit = fit_kahn(probabilities(logits), [0, 0, 0, 1, 1, 1])
    assert fit.weights[0] > 0 > fit.weights[1]
    correct = probabilities([1, -1])
    incorrect = probabilities([-1, 1])
    assert pool_methods(correct)["logodds_sum"].score == pytest.approx(0.5)
    assert pool_methods(incorrect)["logodds_sum"].score == pytest.approx(0.5)
    assert fit.predict(correct).score > 0.5 > fit.predict(incorrect).score


def test_kahn_retains_prior_when_streams_are_uninformative():
    fit = fit_kahn([[0.8, 0.8]] * 10, [1] * 8 + [0] * 2)
    prior = 8.5 / 11
    assert fit.status == "fitted"
    assert np.allclose(fit.weights, 0, atol=1e-8)
    assert fit.predict([0.8, 0.8]).score == pytest.approx(prior)
    assert fit.parameters["prior"] == pytest.approx(prior)


def test_kahn_prior_is_the_posterior_at_the_class_midpoint():
    incorrect = [-2, 0]
    correct = [-1, 0, 1, 1, 2, 3]
    fit = fit_kahn(probabilities([[value] for value in incorrect + correct]), [0] * 2 + [1] * 6)
    midpoint = (np.mean(incorrect) + np.mean(correct)) / 2
    assert fit.predict(probabilities([midpoint])).score == pytest.approx(6.5 / 9)


def test_kahn_is_finite_for_duplicate_streams_and_deterministic_separation():
    for logits in (np.array([-2, -1, 1, 2]), np.array([-1, -1, 1, 1])):
        scores = probabilities(np.column_stack([logits] * 7))
        fit = fit_kahn(scores, [0, 0, 1, 1], shrinkage=0)
        assert np.isfinite(fit.weights).all()
        assert math.isfinite(fit.intercept)
        assert np.linalg.eigvalsh(fit.parameters["regularized_covariance"]).min() > 0
        assert 0 <= fit.predict([0] * 7).score <= 1
        assert 0 <= fit.predict([1] * 7).score <= 1
        json.dumps(fit.parameters, allow_nan=False)


def test_kahn_predictions_are_invariant_to_column_permutation():
    logits = np.array([[-2, -1, 1], [-1, 0, 2], [0, 1, 0], [1, 0, 1], [2, 2, -1], [3, 1, 0]])
    labels = [0, 0, 0, 1, 1, 1]
    permutation = [2, 0, 1]
    fit = fit_kahn(probabilities(logits), labels)
    permuted = fit_kahn(probabilities(logits[:, permutation]), labels)
    target = probabilities([1.2, -0.3, 0.6])
    assert fit.predict(target).score == pytest.approx(permuted.predict(target[permutation]).score)
    assert permuted.weights == pytest.approx(np.asarray(fit.weights)[permutation])


def test_kahn_rejects_bad_inputs_and_reports_insufficient_classes():
    for scores, labels in ((np.empty((0, 2)), []), ([[0.7, 0.8]] * 4, [1] * 4), ([[0.7]] * 3, [0, 1, 1])):
        fit = fit_kahn(scores, labels)
        assert fit.status == "insufficient_classes"
        assert fit.predict([0.7] * fit.n_streams).score is None
    for scores, labels in (([[float("nan")]], [1]), ([[1.1]], [0]), ([[0.5]], [2]), ([[0.5]], [])):
        with pytest.raises(ValueError):
            fit_kahn(scores, labels)
    for shrinkage in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError):
            fit_kahn([[0.2], [0.3], [0.7], [0.8]], [0, 0, 1, 1], shrinkage=shrinkage)
    for ridge in (0, -1, float("inf")):
        with pytest.raises(ValueError):
            fit_kahn([[0.2], [0.3], [0.7], [0.8]], [0, 0, 1, 1], ridge=ridge)
    fit = fit_kahn([[0.2], [0.3], [0.7], [0.8]], [0, 0, 1, 1])
    with pytest.raises(ValueError):
        fit.predict([0.5, 0.5])
