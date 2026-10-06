import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .base import FittedPool, fitting_data


def fit_logistic_pool(scores, labels, l2=1.0):
    logits, outcomes = fitting_data(scores, labels)
    if not np.isfinite(l2) or l2 <= 0:
        raise ValueError("l2 must be finite and positive")
    n_fit, n_streams = logits.shape
    n_errors = int(np.sum(outcomes == 0))
    if min(n_errors, n_fit - n_errors) < 2:
        return FittedPool("logistic_pool", n_streams, None, "insufficient_classes", n_fit, n_errors)
    center = logits.mean(axis=0)
    scale = logits.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    standardized = (logits - center) / scale

    def objective(parameters):
        weights, intercept = parameters[:-1], parameters[-1]
        linear = standardized @ weights + intercept
        residuals = expit(linear) - outcomes
        loss = np.sum(np.logaddexp(0, linear) - outcomes * linear) + l2 * (weights @ weights) / 2
        gradient = np.r_[standardized.T @ residuals + l2 * weights, residuals.sum()]
        return float(loss), gradient

    initial = np.r_[np.zeros(n_streams), np.log((n_fit - n_errors) / n_errors)]
    result = minimize(objective, initial, jac=True, method="L-BFGS-B",
                      options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-7})
    diagnostics = {"l2": float(l2), "objective": "sum_nll_plus_l2_over_2",
                   "optimizer": "L-BFGS-B", "optimizer_message": str(result.message),
                   "iterations": int(result.nit), "converged": bool(result.success)}
    if not result.success or not np.isfinite(result.x).all() or not np.isfinite(result.fun):
        return FittedPool("logistic_pool", n_streams, None, "optimization_failed", n_fit, n_errors, diagnostics)
    weights = result.x[:-1] / scale
    intercept = float(result.x[-1] - weights @ center)
    diagnostics.update(weights=weights.tolist(), intercept=intercept,
                       logit_center=center.tolist(), logit_scale=scale.tolist(),
                       standardized_weights=result.x[:-1].tolist(),
                       objective_value=float(result.fun))
    return FittedPool("logistic_pool", n_streams, None, "fitted", n_fit, n_errors, diagnostics,
                      weights=tuple(weights), intercept=intercept)
