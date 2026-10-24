"""Per-step error scores of every scorer and of the panel, evaluated against ProcessBench first-error labels."""

import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from data.processbench import SPLITS

RUNS = Path(__file__).resolve().parents[2] / "runs" / "cross_scoring"
WINDOWS = (1, 8, 16, 32, None)  # None: mean over the whole step
LENGTH_BINS = 5


def auroc(positive, scores):
    """Probability that a positive outranks a negative, ties counted half (Mann-Whitney)."""
    ranks, n = rankdata(scores), positive.sum()
    return (ranks[positive].sum() - n * (n + 1) / 2) / (n * (len(scores) - n))


def matched_auroc(positive, scores, length):
    """AUROC over pairs of steps of similar length only (length quantile bins, pairs pooled across bins)."""
    bins = np.digitize(length, np.quantile(length, np.linspace(0, 1, LENGTH_BINS + 1)[1:-1]))
    total = pairs = 0.0
    for b in np.unique(bins):
        pos, sel = positive[bins == b], scores[bins == b]
        n = pos.sum() * (len(pos) - pos.sum())
        if n:
            total, pairs = total + auroc(pos, sel) * n, pairs + n
    return total / pairs


def window_scores(saved, width):
    """(solutions, steps): the highest mean surprise over any width-token window inside the step, nan elsewhere."""
    out = np.full((len(saved["labels"]), saved["n_steps"].max()), np.nan)
    surprise, key = -saved["actual"].astype(float), saved["item"].astype(np.int64) * 10_000 + saved["step"]
    bounds = np.flatnonzero(np.diff(key)) + 1
    for segment in np.split(np.arange(len(key)), bounds):
        values = np.concatenate([[0.0], np.cumsum(surprise[segment])])
        w = min(width or len(segment), len(segment))
        out[saved["item"][segment[0]], saved["step"][segment[0]]] = np.max(values[w:] - values[:-w]) / w
    return out


def belief_drops(saved, n_items, n_steps):
    """(solutions, steps): how much the belief in the panel's answer falls across each step, nan if not scored."""
    belief = np.full((n_items, n_steps.max() + 1), np.nan)
    belief[saved["item"], saved["k"]] = saved["belief"]
    return belief[:, :-1] - belief[:, 1:]


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


def evaluate(scores, labels, n_steps, length):
    """Step AUROC, length-matched step AUROC, solution AUROC (largest step score), oracle-threshold F1."""
    scores, position = np.nan_to_num(scores), np.arange(scores.shape[1])[None]
    steps = (position < n_steps[:, None]) & ((labels[:, None] < 0) | (position <= labels[:, None]))
    first = (position == labels[:, None])[steps]
    solution = np.nanmax(np.where(position < n_steps[:, None], scores, np.nan), axis=1)
    return (auroc(first, scores[steps]), matched_auroc(first, scores[steps], length[steps]),
            auroc(labels >= 0, solution), first_error_f1(scores, labels, n_steps))


def report(name, by_scorer, labels, n_steps, length):
    """Panel (median of percentiles) and the best single scorer of one signal."""
    normalised = {s: percentile(v) for s, v in by_scorer.items()}
    rows = {s: evaluate(v, labels, n_steps, length) for s, v in normalised.items()}
    best = max(rows, key=lambda s: rows[s][0])
    panel = evaluate(np.nanmedian(np.stack(list(normalised.values())), axis=0), labels, n_steps, length)
    print(f"  {name:<20} {'panel (median)':<45} " + "  ".join(f"{m:.3f}" for m in panel))
    print(f"  {'':<20} {'best: ' + best:<45} " + "  ".join(f"{m:.3f}" for m in rows[best]))


def main():
    scorers = sorted(p.name for p in (RUNS / "scores").iterdir())
    for split in sys.argv[1:] or SPLITS:
        saved = {s: np.load(RUNS / "scores" / s / f"{split}.npz") for s in scorers}
        labels, n_steps = saved[scorers[0]]["labels"], saved[scorers[0]]["n_steps"]
        common = np.ones(len(labels), dtype=bool)
        for s in scorers:
            common &= np.isin(np.arange(len(labels)), saved[s]["scored"])
        first = saved[scorers[0]]
        length = np.zeros((len(labels), n_steps.max()))
        np.add.at(length, (first["item"], first["step"]), 1)
        labels, n_steps, length = labels[common], n_steps[common], length[common]
        print(f"\n{split}: {common.sum()} solutions ({(labels >= 0).mean():.0%} wrong)   "
              f"step AUROC / length-matched step AUROC / solution AUROC / F1 at best threshold")
        # Controls: first errors tend to sit late and in long steps, whatever the scorer.
        for name, scores in {"step index": np.broadcast_to(np.arange(length.shape[1], dtype=float), length.shape),
                             "step length": length}.items():
            print(f"  {'control':<20} {name:<45} " + "  ".join(f"{m:.3f}"
                                                              for m in evaluate(percentile(scores), labels, n_steps, length)))
        for width in WINDOWS:
            report(f"surprise, {width}-token" if width else "surprise, step mean",{s: window_scores(saved[s], width)[common] for s in scorers},
                   labels, n_steps, length)
        believers = [s for s in scorers if (RUNS / "beliefs" / s / f"{split}.npz").exists()]
        if believers:
            drops = {s: belief_drops(np.load(RUNS / "beliefs" / s / f"{split}.npz"), len(common),
                                     saved[scorers[0]]["n_steps"])[common] for s in believers}
            report("belief drop", drops, labels, n_steps, length)


if __name__ == "__main__":
    main()
