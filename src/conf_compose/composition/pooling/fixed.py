from typing import Optional, Sequence

from ..candidates import logit, sigmoid
from .base import Prediction

FIXED_RULES = ("mean", "logodds_sum", "logodds_mean")


def pool_methods(scores: Sequence[float], prior: Optional[float] = None) -> dict[str, Prediction]:
    if len(scores) == 0:
        return {}
    logits = [logit(value) for value in scores]
    total = sum(logits)
    mean = total / len(logits)
    methods = {
        "mean": Prediction(sum(scores) / len(scores)),
        "logodds_sum": Prediction(sigmoid(total), logit=total),
        "logodds_mean": Prediction(sigmoid(mean), logit=mean),
    }
    if prior is not None:
        methods["prior_logodds_sum"] = Prediction(prior_corrected(logits, prior, 1.0))
        methods["prior_logodds_mean"] = Prediction(prior_corrected(logits, prior, 1 / len(logits)))
    return methods


def prior_corrected(logits: Sequence[float], prior: float, w: float) -> float:
    return sigmoid(w * sum(logits) - (w * len(logits) - 1) * logit(prior))
