"""Share of the 15 groups in which each method beats its solo reference, per dataset: robust to a few large panels."""

import matplotlib.pyplot as plt
import numpy as np

from common import ESTIMATORS, FAMILIES, LABELS, POOLING, TASKS, TASK_NAMES, save, select

METHODS = POOLING + FAMILIES["Ablation"] + FAMILIES["Judge"]


def rate(rows, estimator, method, task, metric):
    selected = select(rows, estimator=estimator, method=method, task=task)
    if not selected:
        return np.nan, 0
    sign = -1 if metric == "ece" else 1
    return float(np.mean([sign * row[f"delta_{metric}"] > 1e-12 for row in selected])), len(selected)


def draw(rows):
    figure, axes = plt.subplots(1, 4, figsize=(15, 5.5), sharey=True)
    panels = [(estimator, metric) for estimator in ESTIMATORS for metric in ("ece", "auarc")]
    for axis, (estimator, metric) in zip(axes, panels):
        grid = np.full((len(METHODS), len(TASKS)), np.nan)
        counts = np.zeros_like(grid, dtype=int)
        for i, method in enumerate(METHODS):
            for j, task in enumerate(TASKS):
                grid[i, j], counts[i, j] = rate(rows, estimator, method, task, metric)
        image = axis.imshow(grid, cmap="RdBu", vmin=0, vmax=1, aspect="auto")
        for i in range(len(METHODS)):
            for j in range(len(TASKS)):
                if counts[i, j]:
                    text = f"{grid[i, j]:.0%}" + ("" if counts[i, j] == 15 else f"\n{counts[i, j]}/15")
                    axis.text(j, i, text, ha="center", va="center", fontsize=6,
                              color="white" if abs(grid[i, j] - 0.5) > 0.35 else "black")
        axis.set_xticks(range(len(TASKS)), [TASK_NAMES[task] for task in TASKS], rotation=40, ha="right")
        axis.set_yticks(range(len(METHODS)), [LABELS[method] for method in METHODS])
        for boundary in (len(POOLING) - 0.5, len(POOLING) + len(FAMILIES["Ablation"]) - 0.5):
            axis.axhline(boundary, color="black", lw=0.8)
        axis.set_title(f"{ESTIMATORS[estimator]}\n{'ECE lower' if metric == 'ece' else 'AUARC higher'} than solo reference")
    figure.colorbar(image, ax=axes, fraction=0.015, label="share of groups improved")
    figure.suptitle("Win rate over the solo reference (blank: no judge run; n/15 marks partial judge coverage)")
    return save(figure, "win_rates")
