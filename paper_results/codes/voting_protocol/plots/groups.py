"""Every one of the 15 groups on every dataset: group x dataset heatmaps of the gain for representative methods."""

import matplotlib.pyplot as plt
import numpy as np

from common import ESTIMATORS, LABELS, TASKS, TASK_NAMES, save, select

METHODS = ["mean", "logodds_sum", "shared_scale", "blp"]
LIMITS = {"ece": 20, "auarc": 10}


def groups(rows):
    """Ordered by size, then by name, so rows of one size sit together."""
    return sorted({(row["n_models"], row["models"]) for row in rows})


def grid(rows, estimator, method, metric, ordered):
    values = np.full((len(ordered), len(TASKS)), np.nan)
    for i, (_, models) in enumerate(ordered):
        for j, task in enumerate(TASKS):
            selected = select(rows, estimator=estimator, method=method, task=task, models=models)
            if selected:
                values[i, j] = 100 * selected[0][f"delta_{metric}"]
    return values


def draw(rows):
    ordered = groups(rows)
    names = [f"{size}: " + models.replace("|", " + ") for size, models in ordered]
    paths = []
    for metric in ("ece", "auarc"):
        figure, axes = plt.subplots(2, len(METHODS), figsize=(16, 10), sharey=True)
        # ECE: blue is a reduction; AUARC: blue is a gain. The colour always means "better".
        cmap = "RdBu_r" if metric == "ece" else "RdBu"
        for row_index, estimator in enumerate(ESTIMATORS):
            for column, method in enumerate(METHODS):
                axis = axes[row_index, column]
                values = grid(rows, estimator, method, metric, ordered)
                image = axis.imshow(values, cmap=cmap, vmin=-LIMITS[metric], vmax=LIMITS[metric], aspect="auto")
                for i in range(values.shape[0]):
                    for j in range(values.shape[1]):
                        axis.text(j, i, f"{values[i, j]:+.1f}", ha="center", va="center", fontsize=6)
                for i in range(1, len(ordered)):
                    if ordered[i][0] != ordered[i - 1][0]:
                        axis.axhline(i - 0.5, color="black", lw=0.8)
                axis.set_xticks(range(len(TASKS)), [TASK_NAMES[task] for task in TASKS], rotation=40, ha="right")
                axis.set_yticks(range(len(names)), names, fontsize=7)
                axis.set_title(f"{LABELS[method]} / {ESTIMATORS[estimator]}", fontsize=9)
        figure.colorbar(image, ax=axes, fraction=0.012,
                        label=f"Delta {metric.upper()} vs solo reference (points; blue is better)")
        figure.suptitle(f"Delta {metric.upper()} for each group (rows: size, then members) and dataset")
        paths += save(figure, f"groups_delta_{metric}")
    return paths
