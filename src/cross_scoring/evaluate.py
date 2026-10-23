"""Per-step error scores of every scorer and of the panel, evaluated against ProcessBench first-error labels."""

import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from data.processbench import SPLITS

SCORES = Path(__file__).resolve().parents[2] / "runs" / "cross_scoring" / "scores"
# Token-level error signals: how unlikely the token is, and how much more likely the scorer's own top token was.
SIGNALS = {"surprise": lambda actual, top: -actual, "disagreement": lambda actual, top: top - actual}


def auroc(positive, scores):
    """Probability that a positive outranks a negative, ties counted half (Mann-Whitney)."""
    ranks, n = rankdata(scores), positive.sum()
    return (ranks[positive].sum() - n * (n + 1) / 2) / (n * (len(scores) - n))


def step_scores(saved, signal):
    """(solutions, steps) array of the largest token score in each step, nan for unscored solutions or padding."""
    out = np.full((len(saved["labels"]), saved["n_steps"].max()), np.nan)
    np.fmax.at(out, (saved["item"], saved["step"]), SIGNALS[signal](saved["actual"], saved["top"]))
    return out


def percentile(scores):
    """Scores mapped to their percentile among all steps of the split, so scorers share one scale."""
    out, valid = np.full(scores.shape, np.nan), ~np.isnan(scores)
    out[valid] = rankdata(scores[valid]) / valid.sum()
    return out


def first_error_f1(scores, labels, n_steps):
    """ProcessBench F1 (harmonic mean of accuracy on wrong and on correct solutions) at the best threshold."""
    best = 0.0
    for threshold in np.linspace(0.5, 1.0, 51):
        above = (scores > threshold) & (np.arange(scores.shape[1]) < n_steps[:, None])
        predicted = np.where(above.any(1), above.argmax(1), -1)
        wrong, correct = np.mean(predicted[labels >= 0] == labels[labels >= 0]), np.mean(predicted[labels < 0] == -1)
        best = max(best, 2 * wrong * correct / max(wrong + correct, 1e-12))
    return best


def evaluate(scores, labels, n_steps):
    """Step AUROC (first wrong step vs steps before it), solution AUROC (largest step score), oracle-threshold F1."""
    scores, position = np.nan_to_num(scores), np.arange(scores.shape[1])[None]
    steps = (position < n_steps[:, None]) & ((labels[:, None] < 0) | (position <= labels[:, None]))
    solution = np.nanmax(np.where(position < n_steps[:, None], scores, np.nan), axis=1)
    return (auroc((position == labels[:, None])[steps], scores[steps]),
            auroc(labels >= 0, solution), first_error_f1(scores, labels, n_steps))


def main():
    scorers = sorted(p.name for p in SCORES.iterdir())
    for split in sys.argv[1:] or SPLITS:
        saved = {s: np.load(SCORES / s / f"{split}.npz") for s in scorers}
        labels, n_steps = saved[scorers[0]]["labels"], saved[scorers[0]]["n_steps"]
        common = np.ones(len(labels), dtype=bool)
        for s in scorers:
            common &= np.isin(np.arange(len(labels)), saved[s]["scored"])
        print(f"\n{split}: {common.sum()} solutions ({(labels[common] >= 0).mean():.0%} wrong)   "
              f"step AUROC / solution AUROC / F1 at best threshold")
        for signal in SIGNALS:
            normalised = {s: percentile(step_scores(saved[s], signal)[common]) for s in scorers}
            normalised["panel (median)"] = np.nanmedian(np.stack(list(normalised.values())), axis=0)
            for name, scores in normalised.items():
                metrics = evaluate(scores, labels[common], n_steps[common])
                print(f"  {signal:<13} {name:<45} " + "  ".join(f"{m:.3f}" for m in metrics))


if __name__ == "__main__":
    main()
