from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from ..candidates import EPSILON, logit, sigmoid


@dataclass
class Prediction:
    score: Optional[float]
    is_probability: bool = True
    answer: Optional[str] = None
    logit: Optional[float] = None
    details: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class FittedPool:
    name: str
    n_streams: int
    scale: Optional[float]
    status: str
    n_fit: int
    n_errors: int
    parameters: dict[str, float] = field(default_factory=dict)

    def predict(self, scores: Sequence[float]) -> Prediction:
        if len(scores) != self.n_streams:
            raise ValueError("prediction must use the fitted number of streams")
        if any(not np.isfinite(value) or not 0 <= value <= 1 for value in scores):
            raise ValueError("scores must be finite probabilities")
        if self.scale is None:
            return Prediction(None)
        pooled_logit = self.scale * sum(logit(value) for value in scores)
        return Prediction(sigmoid(pooled_logit), logit=pooled_logit)


def fitting_data(scores, labels):
    probabilities = np.asarray(scores, dtype=float)
    outcomes = np.asarray(labels, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] < 1:
        raise ValueError("fitting scores must have shape (questions, streams)")
    if outcomes.shape != (len(probabilities),):
        raise ValueError("one correctness label is required per question")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("fitting scores must be finite probabilities")
    if not np.isin(outcomes, (0, 1)).all():
        raise ValueError("correctness labels must be zero or one")
    clipped = np.clip(probabilities, EPSILON, 1 - EPSILON)
    return np.log(clipped) - np.log1p(-clipped), outcomes
