"""Correctness and confidence-quality metrics: accuracy, ECE, Brier, NLL, AUROC, AUARC/AURC."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np

EPS = 1e-6


def _arrays(conf: Sequence[float], correct: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    conf, correct = np.asarray(conf, dtype=float), np.asarray(correct, dtype=float)
    if conf.shape != correct.shape or conf.ndim != 1:
        raise ValueError("conf and correct must be 1-D arrays of equal length")
    if not np.isfinite(conf).all():
        raise ValueError("conf contains NaN or inf")
    return conf, correct


def accuracy(correct: Sequence[float]) -> float:
    return float(np.mean(correct))


def brier(conf: Sequence[float], correct: Sequence[float]) -> float:
    conf, correct = _arrays(conf, correct)
    return float(np.mean((conf - correct) ** 2))


def nll(conf: Sequence[float], correct: Sequence[float]) -> float:
    conf, correct = _arrays(conf, correct)
    p = np.clip(conf, EPS, 1 - EPS)
    return float(-np.mean(correct * np.log(p) + (1 - correct) * np.log(1 - p)))


def ece(conf: Sequence[float], correct: Sequence[float], n_bins: int = 10, adaptive: bool = False) -> float:
    conf, correct = _arrays(conf, correct)
    if adaptive:
        edges = np.quantile(conf, np.linspace(0, 1, n_bins + 1)[1:-1])
        bins = np.searchsorted(edges, conf, side="right")
    else:
        bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    gaps = np.abs(np.bincount(bins, conf - correct, minlength=n_bins))
    return float(gaps.sum() / len(conf))


def auroc(conf: Sequence[float], correct: Sequence[float]) -> float:
    conf, correct = _arrays(conf, correct)
    n_pos = correct.sum()
    n_neg = len(correct) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((_average_ranks(conf)[correct == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def selective_curve(conf: Sequence[float], correct: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Coverage and selective accuracy when keeping the k most confident answers; ties get expected accuracy."""
    conf, correct = _arrays(conf, correct)
    order = np.argsort(-conf, kind="stable")
    sorted_conf, sorted_correct = conf[order], correct[order]
    _, group_start, group_size = np.unique(-sorted_conf, return_index=True, return_counts=True)
    group_mean = np.add.reduceat(sorted_correct, group_start) / group_size
    expected = np.repeat(group_mean, group_size)
    k = np.arange(1, len(conf) + 1)
    return k / len(conf), np.cumsum(expected) / k


def auarc(conf: Sequence[float], correct: Sequence[float]) -> float:
    return float(selective_curve(conf, correct)[1].mean())


def aurc(conf: Sequence[float], correct: Sequence[float]) -> float:
    return 1.0 - auarc(conf, correct)


def oracle_auarc(correct: Sequence[float]) -> float:
    correct = np.asarray(correct, dtype=float)
    return auarc(correct, correct)


def summarize(conf: Sequence[float], correct: Sequence[float], n_bins: int = 10) -> Dict[str, float]:
    return {
        "n": len(correct),
        "accuracy": accuracy(correct),
        "mean_conf": float(np.mean(conf)),
        "ece": ece(conf, correct, n_bins),
        "ace": ece(conf, correct, n_bins, adaptive=True),
        "brier": brier(conf, correct),
        "nll": nll(conf, correct),
        "auroc": auroc(conf, correct),
        "auarc": auarc(conf, correct),
        "oracle_auarc": oracle_auarc(correct),
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    _, start, counts = np.unique(sorted_values, return_index=True, return_counts=True)
    ranks = np.empty(len(values))
    ranks[order] = np.repeat(start + (counts + 1) / 2, counts)
    return ranks
