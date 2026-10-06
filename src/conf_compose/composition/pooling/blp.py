from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import betainc, betaincc

from ..candidates import EPSILON
from .base import FittedPool, Prediction, probability_data

SHAPE_BOUNDS = (0.05, 100.0)


def beta_log_probabilities(values, alpha, beta):
    values = np.clip(values, EPSILON, 1 - EPSILON)
    tiny = np.finfo(float).tiny
    positive = np.log(np.maximum(betainc(alpha, beta, values), tiny))
    negative = np.log(np.maximum(betaincc(alpha, beta, values), tiny))
    return positive, negative


@dataclass(frozen=True)
class FittedBLP(FittedPool):
    alpha: float = 1.0
    beta: float = 1.0

    def predict(self, scores):
        if len(scores) != self.n_streams:
            raise ValueError("prediction must use the fitted number of streams")
        values = np.asarray(scores, dtype=float)
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError("scores must be finite probabilities")
        if self.weights is None:
            return Prediction(None)
        pooled = (sum(scores) / self.n_streams if self.parameters.get("equal_weights")
                  else float(values @ self.weights))
        positive, negative = beta_log_probabilities(pooled, self.alpha, self.beta)
        return Prediction(float(np.exp(positive)), logit=float(positive - negative), ranking_score=pooled)


def fit_blp(scores, labels, equal_weights=False):
    probabilities, outcomes = probability_data(scores, labels)
    n_fit, n_streams = probabilities.shape
    n_errors = int(np.sum(outcomes == 0))
    name = "blp_equal" if equal_weights else "blp"
    if min(n_errors, n_fit - n_errors) < 2:
        return FittedBLP(name, n_streams, None, "insufficient_classes", n_fit, n_errors)
    uniform = np.full(n_streams, 1 / n_streams)
    learn_weights = not equal_weights and n_streams > 1

    def unpack(parameters):
        weights = parameters[:-2] if learn_weights else uniform
        alpha, beta = np.exp(parameters[-2:])
        return weights, alpha, beta

    def objective(parameters):
        weights, alpha, beta = unpack(parameters)
        positive, negative = beta_log_probabilities(probabilities @ weights, alpha, beta)
        return float(-np.mean(outcomes * positive + (1 - outcomes) * negative))

    bounds = ([(0.0, 1.0)] * n_streams if learn_weights else []) + [tuple(np.log(SHAPE_BOUNDS))] * 2
    constraints = ({"type": "eq", "fun": lambda parameters: parameters[:-2].sum() - 1},) if learn_weights else ()
    results = []
    for shape in (1.0, 0.5, 2.0):
        initial = np.r_[uniform if learn_weights else [], np.log([shape, shape])]
        results.append(minimize(objective, initial, method="SLSQP", bounds=bounds, constraints=constraints,
                                options={"maxiter": 500, "ftol": 1e-10}))
    valid = [result for result in results if result.success and np.isfinite(result.fun)
             and np.isfinite(result.x).all()
             and (not learn_weights or abs(result.x[:-2].sum() - 1) < 1e-6)]
    diagnostics = {"equal_weights": bool(equal_weights), "shape_bounds": list(SHAPE_BOUNDS),
                   "optimizer": "SLSQP", "initial_shapes": [1.0, 0.5, 2.0],
                   "successful_starts": len(valid), "messages": [str(result.message) for result in results],
                   "pooled_probability_clip": EPSILON}
    if not valid:
        return FittedBLP(name, n_streams, None, "optimization_failed", n_fit, n_errors, diagnostics)
    result = min(valid, key=lambda value: value.fun)
    weights, alpha, beta = unpack(result.x)
    weights = np.maximum(weights, 0)
    weights /= weights.sum()
    diagnostics.update(weights=weights.tolist(), alpha=float(alpha), beta=float(beta),
                       fit_nll=objective(np.r_[weights if learn_weights else [], np.log([alpha, beta])]),
                       iterations=int(result.nit), converged=True,
                       shape_at_bound=bool(np.any(np.isclose([alpha, beta], SHAPE_BOUNDS[0]))
                                           or np.any(np.isclose([alpha, beta], SHAPE_BOUNDS[1]))))
    return FittedBLP(name, n_streams, None, "fitted", n_fit, n_errors, diagnostics,
                     weights=tuple(weights), alpha=float(alpha), beta=float(beta))
