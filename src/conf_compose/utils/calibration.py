"""Post-hoc calibrators fitted on validation data: Platt (logistic on logit) and beta calibration."""

from __future__ import annotations

from typing import Sequence

import numpy as np

EPS = 1e-6


def _logistic_fit(features: np.ndarray, labels: np.ndarray, l2: float = 1e-4, steps: int = 100) -> np.ndarray:
    design = np.column_stack([features, np.ones(len(features))])
    weights = np.zeros(design.shape[1])
    penalty = l2 * np.eye(design.shape[1])
    penalty[-1, -1] = 0.0
    for _ in range(steps):
        p = 1 / (1 + np.exp(-np.clip(design @ weights, -30, 30)))
        gradient = design.T @ (p - labels) + penalty @ weights
        hessian = design.T @ (design * (p * (1 - p))[:, None]) + penalty
        step = np.linalg.solve(hessian + 1e-9 * np.eye(len(weights)), gradient)
        weights -= step
        if np.abs(step).max() < 1e-8:
            break
    return weights


class Calibrator:
    def __init__(self):
        self.weights = None

    def fit(self, conf: Sequence[float], correct: Sequence[float]) -> "Calibrator":
        self.weights = _logistic_fit(self.features(np.asarray(conf, dtype=float)), np.asarray(correct, dtype=float))
        return self

    def predict(self, conf: Sequence[float]) -> np.ndarray:
        design = np.column_stack([self.features(np.asarray(conf, dtype=float)), np.ones(len(conf))])
        return 1 / (1 + np.exp(-np.clip(design @ self.weights, -30, 30)))

    @staticmethod
    def features(conf: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class PlattCalibrator(Calibrator):
    @staticmethod
    def features(conf: np.ndarray) -> np.ndarray:
        p = np.clip(conf, EPS, 1 - EPS)
        return np.log(p / (1 - p))[:, None]


class BetaCalibrator(Calibrator):
    @staticmethod
    def features(conf: np.ndarray) -> np.ndarray:
        p = np.clip(conf, EPS, 1 - EPS)
        return np.column_stack([np.log(p), -np.log(1 - p)])


CALIBRATORS = {"platt": PlattCalibrator, "beta": BetaCalibrator}
