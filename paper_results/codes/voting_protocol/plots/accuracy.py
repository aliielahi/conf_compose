"""Separate the AUARC gain the vote gets from higher accuracy from any gain in how well confidence ranks answers."""

import matplotlib.pyplot as plt
import numpy as np

from common import (COLORS, ESTIMATORS, FAMILIES, LABELS, POOLING, TASKS, TASK_NAMES, bootstrap_mean, ranking_skill,
                    save, select)

METHODS = POOLING + FAMILIES["Ablation"]
TASK_COLORS = dict(zip(TASKS, ("#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e")))


def skill_gain(row):
    """Ranking skill of the pooled score on the vote's labels minus that of the solo reference on its own labels."""
    pooled = ranking_skill(row["auarc"], row["accuracy"], row["n_matched"])
    solo = ranking_skill(row["reference_auarc"], row["reference_auarc_accuracy"], row["n_matched"])
    return 100 * (pooled - solo)


def draw(rows):
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1, 1, 1.3]})
    for axis, estimator in zip(axes[:2], ESTIMATORS):
        selected = select(rows, estimator=estimator, method="mean")
        gains = np.array([100 * (row["accuracy"] - row["reference_auarc_accuracy"]) for row in selected])
        deltas = np.array([100 * row["delta_auarc"] for row in selected])
        for task in TASKS:
            mask = np.array([row["task"] == task for row in selected])
            axis.scatter(gains[mask], deltas[mask], s=18, color=TASK_COLORS[task], label=TASK_NAMES[task])
        limits = [min(gains.min(), deltas.min()) - 1, max(gains.max(), deltas.max()) + 1]
        axis.plot(limits, limits, color="#999999", lw=0.8, ls="--")
        correlation = np.corrcoef(gains, deltas)[0, 1]
        axis.set_title(f"Mean pooling / {ESTIMATORS[estimator]}  (r = {correlation:.2f})")
        axis.set_xlabel("Vote accuracy minus reference accuracy (points)")
        axis.set_ylabel("Delta AUARC (points)")
        axis.legend(frameon=False, fontsize=7)
    axis = axes[2]
    width = 0.38
    positions = np.arange(len(METHODS))
    for offset, (estimator, title) in zip((-width / 2, width / 2), ESTIMATORS.items()):
        values = [bootstrap_mean([skill_gain(row) for row in select(rows, estimator=estimator, method=method)])
                  for method in METHODS]
        means = np.array([value[0] for value in values])
        errors = np.array([[value[0] - value[1] for value in values], [value[2] - value[0] for value in values]])
        axis.bar(positions + offset, means, width, yerr=errors, capsize=2, label=title,
                 color="#4c72b0" if estimator == "cons" else "#dd8452")
    axis.axhline(0, color="#999999", lw=0.8)
    axis.set_xticks(positions, [LABELS[method] for method in METHODS], rotation=45, ha="right")
    axis.set_ylabel("Ranking skill gain (points of the oracle gap)")
    axis.set_title("AUARC with accuracy removed: 0 = random, 100 = oracle ranking")
    axis.legend(frameon=False)
    figure.suptitle("Is the AUARC gain from a better vote or from better confidence ranking?")
    figure.tight_layout()
    return save(figure, "accuracy_vs_ranking")
