"""Average rank of each method across all 75 panels, the descriptive view behind a Friedman comparison."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

from common import COLORS, ESTIMATORS, FAMILIES, LABELS, POOLING, save, select

METHODS = ["best_solo"] + POOLING + FAMILIES["Ablation"]


def average_ranks(rows, estimator, metric):
    """Rank 1 is best in every panel; methods with identical scores share the average rank."""
    panels = sorted({(row["task"], row["models"]) for row in select(rows, estimator=estimator)})
    table = []
    for task, models in panels:
        values = {row["method"]: row[metric] for row in select(rows, estimator=estimator, task=task, models=models)}
        if all(method in values for method in METHODS):
            scores = np.array([values[method] for method in METHODS])
            table.append(rankdata(scores if metric == "ece" else -np.round(scores, 10)))
    return np.mean(table, axis=0), len(table)


def draw(rows):
    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    for column, estimator in enumerate(ESTIMATORS):
        for row_index, metric in enumerate(("ece", "auarc")):
            axis = axes[row_index, column]
            ranks, n = average_ranks(rows, estimator, metric)
            order = np.argsort(ranks)
            positions = np.arange(len(METHODS))
            axis.barh(positions, ranks[order], color=[COLORS[METHODS[i]] for i in order])
            axis.set_yticks(positions, [LABELS[METHODS[i]] for i in order])
            axis.invert_yaxis()
            for position, value in zip(positions, ranks[order]):
                axis.text(value + 0.1, position, f"{value:.1f}", va="center", fontsize=7)
            axis.set_xlim(0, len(METHODS) + 1)
            axis.set_xlabel(f"average rank over {n} panels (1 is best)")
            axis.set_title(f"{ESTIMATORS[estimator]} / {metric.upper()}")
    figure.suptitle("Average rank per panel. Equal AUARC ranks reveal methods that order questions identically.")
    figure.tight_layout()
    return save(figure, "average_ranks")
