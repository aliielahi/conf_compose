"""Consistency against sequence probability as the members' input, panel by panel, before and after pooling."""

import matplotlib.pyplot as plt
import numpy as np

from common import TASKS, TASK_NAMES, save, select
from accuracy import TASK_COLORS

METHODS = {"best_solo": "Solo reference", "mean": "Mean", "blp": "Weighted BLP"}


def paired(rows, method, metric):
    """Absolute metric under each estimator for the same dataset and group."""
    pairs = []
    for row in select(rows, estimator="cons", method=method):
        other = select(rows, estimator="seq", method=method, task=row["task"], models=row["models"])
        if other:
            pairs.append((row["task"], 100 * row[metric], 100 * other[0][metric]))
    return pairs


def draw(rows):
    figure, axes = plt.subplots(2, len(METHODS), figsize=(12, 7.5))
    for row_index, metric in enumerate(("ece", "auarc")):
        for column, (method, title) in enumerate(METHODS.items()):
            axis = axes[row_index, column]
            pairs = paired(rows, method, metric)
            for task in TASKS:
                points = np.array([(x, y) for name, x, y in pairs if name == task])
                axis.scatter(points[:, 0], points[:, 1], s=16, color=TASK_COLORS[task], label=TASK_NAMES[task])
            values = np.array([(x, y) for _, x, y in pairs])
            limits = [values.min() - 1, values.max() + 1]
            axis.plot(limits, limits, color="#999999", lw=0.8, ls="--")
            better = np.mean(values[:, 1] < values[:, 0]) if metric == "ece" else np.mean(values[:, 1] > values[:, 0])
            axis.set_title(f"{title} {metric.upper()}: sequence better in {better:.0%} of panels", fontsize=9)
            axis.set_xlabel(f"{metric.upper()} with consistency (points)")
            axis.set_ylabel(f"{metric.upper()} with sequence probability (points)")
            if row_index == 0 and column == 0:
                axis.legend(frameon=False, fontsize=7)
    figure.suptitle("Which member confidence to pool? Each point is one group on one dataset "
                    "(masks and references are matched within, not across, estimators)")
    figure.tight_layout()
    return save(figure, "estimators")
