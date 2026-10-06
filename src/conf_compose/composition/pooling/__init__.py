from functools import partial

from .base import FittedPool, Prediction
from .blp import FittedBLP, fit_blp
from .fixed import FIXED_RULES, pool_methods, prior_corrected
from .kahn import fit_kahn
from .logistic import fit_logistic_pool
from .shared_rho import fit_shared_rho
from .shared_scale import fit_shared_scale

LEARNED_FITTERS = {"shared_rho": fit_shared_rho, "shared_scale": fit_shared_scale, "kahn": fit_kahn,
                   "blp": fit_blp, "logistic_pool": fit_logistic_pool}
ABLATION_FITTERS = {"blp_equal": partial(fit_blp, equal_weights=True),
                    "kahn_diagonal": partial(fit_kahn, covariance_mode="diagonal")}
RULES = (*FIXED_RULES, *LEARNED_FITTERS)


def fit_methods(scores, labels, ablations=False, logistic_l2=1.0):
    fitters = {**LEARNED_FITTERS, "logistic_pool": partial(fit_logistic_pool, l2=logistic_l2)}
    if ablations:
        fitters.update(ABLATION_FITTERS)
    return {name: fitter(scores, labels) for name, fitter in fitters.items()}


__all__ = ["FittedPool", "FittedBLP", "Prediction", "FIXED_RULES", "LEARNED_FITTERS", "RULES",
           "pool_methods", "prior_corrected", "fit_shared_rho", "fit_shared_scale", "fit_kahn",
           "fit_blp", "fit_logistic_pool", "fit_methods"]
