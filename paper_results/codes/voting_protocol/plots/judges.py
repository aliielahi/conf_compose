"""Judges against pooling on the panels where every judge row exists, and what showing confidences does to a judge."""

import matplotlib.pyplot as plt
import numpy as np

from common import COLORS, ESTIMATORS, FAMILIES, LABELS, TASK_NAMES, save, select

JUDGES = FAMILIES["Judge"]
COMPARED = ["best_solo", "mean", "shared_scale", "blp"] + JUDGES


def common_panels(rows, estimator):
    """Panels with all four judge rows, so judges and pooling are averaged over the same groups."""
    panels = {(row["task"], row["models"]) for row in select(rows, estimator=estimator)}
    return sorted(panel for panel in panels
                  if all(select(rows, estimator=estimator, task=panel[0], models=panel[1], method=method)
                         for method in JUDGES))


def value(rows, estimator, method, task, models, metric):
    return 100 * select(rows, estimator=estimator, method=method, task=task, models=models)[0][metric]


def draw(rows):
    figure, axes = plt.subplots(2, 2, figsize=(13, 8))
    for column, estimator in enumerate(ESTIMATORS):
        panels = common_panels(rows, estimator)
        if not panels:
            raise ValueError("No matched judge panels in atomic.csv; regenerate the tables with the correct --judge-dir first")
        tasks = sorted({task for task, _ in panels}, key=list(TASK_NAMES).index)
        for row_index, metric in enumerate(("ece", "auarc")):
            axis = axes[row_index, column]
            width = 0.8 / len(COMPARED)
            for index, method in enumerate(COMPARED):
                means = [np.mean([value(rows, estimator, method, task, models, metric)
                                  for name, models in panels if name == task]) for task in tasks]
                axis.bar(np.arange(len(tasks)) + (index - len(COMPARED) / 2 + 0.5) * width, means, width,
                         color=COLORS[method], label=LABELS[method])
            counts = [sum(name == task for name, _ in panels) for task in tasks]
            axis.set_xticks(range(len(tasks)), [f"{TASK_NAMES[task]}\n({n} groups)" for task, n in zip(tasks, counts)])
            axis.set_ylabel(f"{metric.upper()} (points, {'lower' if metric == 'ece' else 'higher'} is better)")
            axis.set_title(f"{ESTIMATORS[estimator]} / {metric.upper()}")
            if metric == "auarc":
                low = min(bar.get_height() for bar in axis.patches)
                axis.set_ylim(max(0, low - 10), None)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=8, frameon=False, bbox_to_anchor=(0.5, -0.03))
    figure.suptitle("Judges vs pooling on identical model groups and matched questions")
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    return save(figure, "judges") + draw_confidence_effect(rows)


def draw_confidence_effect(rows):
    """Per group: the same judge reading reasoning only, then reasoning plus the members' confidences."""
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), sharey="row")
    for column, estimator in enumerate(ESTIMATORS):
        for row_index, metric in enumerate(("ece", "auarc")):
            axis = axes[row_index, column]
            panels = common_panels(rows, estimator)
            means = []
            for offset, judge in enumerate(("g3-27i", "l32-3bi")):
                plain, shown = f"judge:{judge}:reasoning", f"judge:{judge}:reasoning_confidence"
                deltas = []
                for task, models in panels:
                    before = value(rows, estimator, plain, task, models, metric)
                    after = value(rows, estimator, shown, task, models, metric)
                    axis.plot([2 * offset, 2 * offset + 1], [before, after], color=COLORS[shown], alpha=0.35, lw=0.8)
                    deltas.append(after - before)
                means.append(np.mean(deltas))
            axis.set_xticks(range(4), ["27B", f"27B + conf\n(mean {means[0]:+.2f})",
                                       "3B", f"3B + conf\n(mean {means[1]:+.2f})"])
            axis.set_ylabel(f"{metric.upper()} (points)")
            axis.set_title(f"{ESTIMATORS[estimator]} / {metric.upper()}")
    figure.suptitle("Showing the members' confidences to the judge, per group (lines connect the same group and dataset)")
    figure.tight_layout()
    return save(figure, "judge_confidence_effect")
