from .base import FittedPool, Prediction
from .fixed import FIXED_RULES, pool_methods, prior_corrected
from .shared_rho import fit_shared_rho
from .shared_scale import fit_shared_scale

LEARNED_FITTERS = {"shared_rho": fit_shared_rho, "shared_scale": fit_shared_scale}
RULES = (*FIXED_RULES, *LEARNED_FITTERS)

__all__ = ["FittedPool", "Prediction", "FIXED_RULES", "LEARNED_FITTERS", "RULES",
           "pool_methods", "prior_corrected", "fit_shared_rho", "fit_shared_scale"]
