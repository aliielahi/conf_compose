import math
from typing import Optional

import numpy as np

from .base import FittedPool, fitting_data


def fit_kahn(scores, labels, shrinkage: Optional[float] = None, ridge: float = 1e-6,
             covariance_mode: str = "full") -> FittedPool:
    logits, outcomes = fitting_data(scores, labels)
    if covariance_mode not in ("full", "diagonal"):
        raise ValueError("covariance_mode must be full or diagonal")
    name = "kahn" if covariance_mode == "full" else "kahn_diagonal"
    if shrinkage is not None and (not np.isfinite(shrinkage) or not 0 <= shrinkage <= 1):
        raise ValueError("shrinkage must be None or a number between zero and one")
    if not np.isfinite(ridge) or ridge <= 0:
        raise ValueError("ridge must be finite and positive")
    n_fit, n_streams = logits.shape
    n_errors = int(np.sum(outcomes == 0))
    if min(n_errors, n_fit - n_errors) < 2:
        return FittedPool(name, n_streams, None, "insufficient_classes", n_fit, n_errors)

    mean_incorrect = logits[outcomes == 0].mean(axis=0)
    mean_correct = logits[outcomes == 1].mean(axis=0)
    residuals = logits - np.where(outcomes[:, None] == 1, mean_correct, mean_incorrect)
    degrees_of_freedom = n_fit - 2
    covariance = residuals.T @ residuals / degrees_of_freedom
    amount = _oas_shrinkage(covariance, degrees_of_freedom) if shrinkage is None else float(shrinkage)
    variance = float(np.trace(covariance) / n_streams)
    ridge_amount = ridge * max(variance, 1.0)
    regularized = (1 - amount) * covariance + (amount * variance + ridge_amount) * np.eye(n_streams)
    if covariance_mode == "diagonal":
        regularized = np.diag(np.diag(regularized))
    weights = np.linalg.solve(regularized, mean_correct - mean_incorrect)
    prior = (n_fit - n_errors + 0.5) / (n_fit + 1)
    intercept = math.log(prior / (1 - prior)) - float(weights @ ((mean_correct + mean_incorrect) / 2))
    parameters = {
        "weights": weights.tolist(),
        "intercept": intercept,
        "prior": prior,
        "prior_pseudocount": 0.5,
        "mean_correct": mean_correct.tolist(),
        "mean_incorrect": mean_incorrect.tolist(),
        "covariance": covariance.tolist(),
        "covariance_mode": covariance_mode,
        "regularized_covariance": regularized.tolist(),
        "covariance_degrees_of_freedom": degrees_of_freedom,
        "shrinkage": amount,
        "shrinkage_estimator": "oas" if shrinkage is None else "fixed",
        "shrinkage_target": "scaled_identity",
        "ridge": ridge,
        "ridge_amount": ridge_amount,
    }
    return FittedPool(name, n_streams, None, "fitted", n_fit, n_errors, parameters,
                      weights=tuple(float(value) for value in weights), intercept=intercept)


def _oas_shrinkage(covariance, degrees_of_freedom):
    dimension = len(covariance)
    trace = float(np.trace(covariance))
    squared_trace = float(np.sum(covariance * covariance))
    denominator = (degrees_of_freedom + 1 - 2 / dimension) * (squared_trace - trace * trace / dimension)
    if denominator <= 0:
        return 1.0
    numerator = (1 - 2 / dimension) * squared_trace + trace * trace
    return float(np.clip(numerator / denominator, 0.0, 1.0))
