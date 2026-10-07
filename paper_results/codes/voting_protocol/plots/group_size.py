"""How each method's gain over the solo reference changes as the group grows from 2 to 6 models."""

import matplotlib.pyplot as plt
import numpy as np

from common import COLORS, ESTIMATORS, LABELS, MARKERS, POOLING, SIZES, TASKS, TASK_NAMES, family, save, select

METRICS = {"ece": "Delta ECE (points, lower is better)", "auarc": "Delta AUARC (points, higher is better)"}


def by_size(rows, estimator, method, metric, task=None):
    means = []
    for size in SIZES:
        selected = [row for row in select(rows, estimator=estimator, method=method, n_models=size)
                    if task is None or row["task"] == task]
        means.append(100 * np.mean([row[f"delta_{metric}"] for row in selected]) if selected else np.nan)
    return means


def draw(rows):
    figure, axes = plt.subplots(2, 2, figsize=(9, 6.5), sharex=True)
    for column, (estimator, title) in enumerate(ESTIMATORS.items()):
        for row_index, (metric, label) in enumerate(METRICS.items()):
            axis = axes[row_index, column]
            axis.axhline(0, color="#999999", lw=0.8)
            for method in POOLING:
                axis.plot(SIZES, by_size(rows, estimator, method, metric), marker=MARKERS[family(method)],
                          color=COLORS[method], lw=2.2 if method in ("logodds_sum", "blp") else 1.2,
                          ms=4, label=LABELS[method])
            axis.set_title(f"{title}")
            axis.set_ylabel(label)
            axis.set_xticks(SIZES)
            if row_index == 1:
                axis.set_xlabel("Models in the group")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.07))
    counts = ", ".join(f"{size}: {len({row['models'] for row in select(rows, n_models=size)})}" for size in SIZES)
    figure.suptitle(f"Gain over the solo reference by group size (groups per size: {counts})")
    paths = save(figure, "group_size")
    return paths + draw_per_task(rows)


def draw_per_task(rows):
    """The same curves for one representative of each family, split by dataset, so one task cannot drive the mean."""
    methods = ["mean", "logodds_sum", "shared_scale", "blp"]
    paths = []
    for metric, label in METRICS.items():
        figure, axes = plt.subplots(2, len(TASKS), figsize=(14, 5.5), sharex=True)
        for row_index, (estimator, title) in enumerate(ESTIMATORS.items()):
            for column, task in enumerate(TASKS):
                axis = axes[row_index, column]
                axis.axhline(0, color="#999999", lw=0.8)
                for method in methods:
                    axis.plot(SIZES, by_size(rows, estimator, method, metric, task), marker=MARKERS[family(method)],
                              color=COLORS[method], lw=1.4, ms=4, label=LABELS[method])
                axis.set_title(f"{TASK_NAMES[task]} / {title}", fontsize=8)
                axis.set_xticks(SIZES)
                if column == 0:
                    axis.set_ylabel(label)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.04))
        figure.suptitle(f"{label.split(' (')[0]} by group size, per dataset")
        paths += save(figure, f"group_size_{metric}_per_task")
    return paths
