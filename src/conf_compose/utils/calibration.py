"""Post-hoc calibrators fitted on validation data: Platt (logistic on logit) and monotone beta calibration."""

from __future__ import annotations

from typing import Sequence

import numpy as np

EPS = 1e-3


def _loss(design: np.ndarray, labels: np.ndarray, weights: np.ndarray, l2: float) -> float:
    z = design @ weights
    return float(np.sum(np.logaddexp(0, z) - labels * z) + 0.5 * l2 * np.sum(weights[:-1] ** 2))


def _logistic_fit(features: np.ndarray, labels: np.ndarray, l2: float = 1e-2, steps: int = 100) -> np.ndarray:
    design = np.column_stack([features, np.ones(len(features))])
    weights = np.zeros(design.shape[1])
    penalty = l2 * np.eye(design.shape[1])
    penalty[-1, -1] = 0.0
    loss = _loss(design, labels, weights, l2)
    for _ in range(steps):
        p = 1 / (1 + np.exp(-np.clip(design @ weights, -30, 30)))
        gradient = design.T @ (p - labels) + penalty @ weights
        hessian = design.T @ (design * (p * (1 - p))[:, None]) + penalty + 1e-9 * np.eye(len(weights))
        step = np.linalg.solve(hessian, gradient)
        scale = 1.0
        while scale > 1e-8 and _loss(design, labels, weights - scale * step, l2) > loss:
            scale /= 2
        if scale <= 1e-8:
            break
        weights = weights - scale * step
        new_loss = _loss(design, labels, weights, l2)
        if loss - new_loss < 1e-10:
            break
        loss = new_loss
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
    def fit(self, conf: Sequence[float], correct: Sequence[float]) -> "BetaCalibrator":
        features = self.features(np.asarray(conf, dtype=float))
        labels = np.asarray(correct, dtype=float)
        weights = _logistic_fit(features, labels)
        for column in (0, 1):
            if weights[column] < 0:
                keep = [c for c in (0, 1) if c != column]
                reduced = _logistic_fit(features[:, keep], labels)
                weights = np.zeros(3)
                weights[keep + [2]] = reduced
                weights[keep] = np.maximum(weights[keep], 0)
                break
        self.weights = weights
        return self

    @staticmethod
    def features(conf: np.ndarray) -> np.ndarray:
        p = np.clip(conf, EPS, 1 - EPS)
        return np.column_stack([np.log(p), -np.log(1 - p)])


CALIBRATORS = {"platt": PlattCalibrator, "beta": BetaCalibrator}


def temperature_scale(conf: Sequence[float], temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    probabilities = np.clip(np.asarray(conf, dtype=float), 1e-6, 1 - 1e-6)
    logits = np.log(probabilities) - np.log1p(-probabilities)
    scaled = np.clip(logits / temperature, -700, 700)
    return 1 / (1 + np.exp(-scaled))


def fit_temperature(conf: Sequence[float], correct: Sequence[float]) -> float:
    probabilities = np.asarray(conf, dtype=float)
    labels = np.asarray(correct, dtype=float)
    if probabilities.ndim != 1 or probabilities.shape != labels.shape or not len(labels):
        raise ValueError("temperature fitting needs nonempty, paired 1-D arrays")
    if not np.isfinite(probabilities).all() or not np.isfinite(labels).all():
        raise ValueError("temperature fitting received nonfinite values")
    logits = np.log(np.clip(probabilities, 1e-6, 1 - 1e-6)) - np.log1p(-np.clip(probabilities, 1e-6, 1 - 1e-6))

    def loss(log_temperature: float) -> float:
        scaled = logits * np.exp(-log_temperature)
        return float(np.mean(np.logaddexp(0, scaled) - labels * scaled))

    left, right = np.log(0.05), np.log(20.0)
    ratio = (np.sqrt(5) - 1) / 2
    a, b = right - ratio * (right - left), left + ratio * (right - left)
    for _ in range(80):
        if loss(a) < loss(b):
            right, b = b, a
            a = right - ratio * (right - left)
        else:
            left, a = a, b
            b = left + ratio * (right - left)
    chosen = min((left, (left + right) / 2, right, 0.0), key=loss)
    return float(np.exp(chosen))
