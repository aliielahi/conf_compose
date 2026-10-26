"""Belief b_t against value V_t along tracked traces, and label-free movement signals that flag wrong traces.

b_t: share of forced answers that are correct; V_t: share of rollouts that end correct; c_t: share of forced answers
equal to the trace's own final answer (needs no label).
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from data.answers import equal, normalise

RUNS = Path(__file__).resolve().parents[2] / "runs" / "belief_tracking"


def share(answers, target):
    return np.mean([equal(a, target) for a in answers])


def tracked(records):
    """One row per tracked trace: question, correctness, length, self-consistency and its b, V, c over cuts."""
    rows = []
    for q, record in enumerate(records):
        votes = Counter(normalise(t["answer"]) for t in record["traces"])
        for t in record["tracked"]:
            own = record["traces"][t["trace"]]["answer"]
            cuts = sorted(t["cuts"], key=lambda c: c["position"])
            rows.append({"question": q, "correct": equal(own, record["answer"]), "answered": own is not None,
                         "length": t["thinking_tokens"],
                         "consistency": (votes[normalise(own)] - 1) / (len(record["traces"]) - 1) if own else 0.0,
                         "b": np.array([share(c["forced"], record["answer"]) for c in cuts]),
                         "V": np.array([share(c["rollouts"], record["answer"]) for c in cuts]),
                         "c": np.array([share(c["forced"], own) for c in cuts]),
                         "n_forced": len(cuts[0]["forced"]), "n_rollouts": len(cuts[0]["rollouts"])})
    return rows


def excess_movement(x):
    """Augenblick-Rabin: squared movement minus the uncertainty it resolved; zero in expectation for a Bayesian."""
    return np.sum(np.diff(x) ** 2) - (x[0] * (1 - x[0]) - x[-1] * (1 - x[-1]))


def auroc(positive, scores):
    ranks, n = rankdata(scores), positive.sum()
    return (ranks[positive].sum() - n * (n + 1) / 2) / (n * (len(scores) - n))


def within_auroc(positive, scores, groups):
    """AUROC over (positive, negative) pairs of the same question only, ties counted half."""
    wins = pairs = 0.0
    for g in np.unique(groups):
        pos, neg = scores[(groups == g) & positive], scores[(groups == g) & ~positive]
        wins += (pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()
        pairs += len(pos) * len(neg)
    return wins / pairs if pairs else np.nan


def calibration(rows):
    """How well the forced-answer belief b_t matches the rollout value V_t, pooled and per cut."""
    b, V = np.stack([r["b"] for r in rows]), np.stack([r["V"] for r in rows])
    nb, nV = rows[0]["n_forced"], rows[0]["n_rollouts"]
    noise = np.mean(b * (1 - b) / (nb - 1) + V * (1 - V) / (nV - 1))
    print(f"  b vs V: correlation {np.corrcoef(b.ravel(), V.ravel())[0, 1]:.3f}, mean squared gap "
          f"{np.mean((b - V) ** 2):.3f} (sampling noise alone {noise:.3f}), mean b - V {np.mean(b - V):+.3f}")
    print(f"  {'cut':>5} {'mean b':>8} {'mean V':>8} {'corr':>7}")
    for t in range(b.shape[1]):
        corr = np.corrcoef(b[:, t], V[:, t])[0, 1] if b[:, t].std() and V[:, t].std() else np.nan
        print(f"  {t:>5} {b[:, t].mean():>8.3f} {V[:, t].mean():>8.3f} {corr:>7.3f}")
    print(f"  {'b bin':>9} {'n':>6} {'mean V':>8}")
    bins = np.minimum((b.ravel() * 5).astype(int), 4)
    for k in range(5):
        if (bins == k).any():
            print(f"  {k / 5:.1f}-{(k + 1) / 5:.1f} {(bins == k).sum():>6} {V.ravel()[bins == k].mean():>8.3f}")


def martingale(rows):
    """Increment regressed on level: slope and intercept are both zero for a martingale."""
    for name in ("b", "V", "c"):
        level = np.concatenate([r[name][:-1] for r in rows])
        step = np.concatenate([np.diff(r[name]) for r in rows])
        slope, intercept = np.polyfit(level, step, 1)
        moves = np.array([excess_movement(r[name]) for r in rows])
        print(f"  {name}: increment = {intercept:+.3f} {slope:+.3f} * level;   mean excess movement {moves.mean():+.3f}")


def detection(rows):
    """AUROC for wrong traces, over all traces and within questions, of label-free signals and controls."""
    wrong = np.array([not r["correct"] for r in rows])
    groups = np.array([r["question"] for r in rows])
    signals = {"control: thinking length": [r["length"] for r in rows],
               "control: 1 - self-consistency (16 traces)": [1 - r["consistency"] for r in rows],
               "1 - mean c before the end": [1 - r["c"][:-1].mean() for r in rows],
               "excess movement of c": [excess_movement(r["c"]) for r in rows],
               "excess movement of b (uses the label)": [excess_movement(r["b"]) for r in rows]}
    print(f"  {'signal':<45} {'AUROC':>7} {'within':>7}")
    for name, scores in signals.items():
        scores = np.array(scores, dtype=float)
        print(f"  {name:<45} {auroc(wrong, scores):>7.3f} {within_auroc(wrong, scores, groups):>7.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    args = parser.parse_args()
    for path in sorted((RUNS / args.model.replace("/", "__")).glob("*.json")):
        rows = tracked(json.loads(path.read_text()))
        mixed = len({r["question"] for r in rows if not r["correct"]} & {r["question"] for r in rows if r["correct"]})
        print(f"\n{path.stem}: {len(rows)} tracked traces, {np.mean([r['correct'] for r in rows]):.0%} correct, "
              f"{np.mean([not r['answered'] for r in rows]):.0%} unanswered, {mixed} questions with both outcomes")
        print(" calibration")
        calibration(rows)
        print(" martingale")
        martingale(rows)
        print(" wrong-trace detection")
        detection(rows)


if __name__ == "__main__":
    main()
