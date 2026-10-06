import numpy as np

from .base import FittedPool, fitting_data


def fit_shared_rho(scores, labels) -> FittedPool:
    logits, outcomes = fitting_data(scores, labels)
    n_fit, n_streams = logits.shape
    n_errors = int(np.sum(outcomes == 0))
    if n_streams == 1:
        return FittedPool("shared_rho", 1, 1.0, "single_stream", n_fit, n_errors)
    if min(n_errors, n_fit - n_errors) < 2:
        return FittedPool("shared_rho", n_streams, None, "insufficient_classes", n_fit, n_errors)

    residuals = logits.copy()
    for outcome in (0, 1):
        mask = outcomes == outcome
        residuals[mask] -= logits[mask].mean(axis=0)
    covariance = residuals.T @ residuals / (n_fit - 2)
    deviations = np.sqrt(np.diag(covariance))
    if np.any(deviations <= 1e-12):
        return FittedPool("shared_rho", n_streams, None, "zero_variance", n_fit, n_errors)

    correlation = covariance / np.outer(deviations, deviations)
    raw_rho = float(correlation[np.triu_indices(n_streams, k=1)].mean())
    rho = float(np.clip(raw_rho, 0.0, 1.0))
    scale = 1 / (1 + (n_streams - 1) * rho)
    return FittedPool("shared_rho", n_streams, scale, "fitted", n_fit, n_errors,
                      {"rho": rho, "raw_rho": raw_rho})
