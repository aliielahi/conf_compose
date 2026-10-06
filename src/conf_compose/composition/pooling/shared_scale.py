import numpy as np

from .base import FittedPool, fitting_data


def fit_shared_scale(scores, labels, bounds=(1e-3, 100.0)) -> FittedPool:
    logits, outcomes = fitting_data(scores, labels)
    n_fit, n_streams = logits.shape
    n_errors = int(np.sum(outcomes == 0))
    lower, upper = bounds
    if not np.isfinite(bounds).all() or not 0 < lower < upper:
        raise ValueError("scale bounds must be finite, positive and increasing")
    parameters = {"lower_bound": float(lower), "upper_bound": float(upper)}
    if min(n_errors, n_fit - n_errors) < 1:
        return FittedPool("shared_scale", n_streams, None, "insufficient_classes", n_fit, n_errors, parameters)
    totals = logits.sum(axis=1)
    if np.all(np.abs(totals) <= 1e-12):
        return FittedPool("shared_scale", n_streams, None, "zero_information", n_fit, n_errors, parameters)

    def derivative(scale):
        values = scale * totals
        probabilities = np.exp(-np.logaddexp(0, -values))
        return float(np.mean(totals * (probabilities - outcomes)))

    if derivative(lower) >= 0:
        scale, status = lower, "lower_bound"
    elif derivative(upper) <= 0:
        scale, status = upper, "upper_bound"
    else:
        for _ in range(80):
            midpoint = (lower + upper) / 2
            if derivative(midpoint) > 0:
                upper = midpoint
            else:
                lower = midpoint
        scale, status = (lower + upper) / 2, "fitted"
    return FittedPool("shared_scale", n_streams, float(scale), status, n_fit, n_errors, parameters)
