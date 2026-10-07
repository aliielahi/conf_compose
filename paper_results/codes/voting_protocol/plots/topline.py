"""Topline: every pooling method's mean change in ECE and AUARC over the solo reference, for both estimators."""

import matplotlib.pyplot as plt
import numpy as np

from common import ESTIMATORS, FAMILIES, LABELS, POOLING, bootstrap_mean, family, save, select

METHODS = POOLING + FAMILIES["Ablation"]
ESTIMATOR_STYLE = {"cons": ("#4c72b0", "o"), "seq": ("#dd8452", "s")}


def draw(rows):
    figure, axes = plt.subplots(1, 2, figsize=(9, 4.8), sharey=True)
    positions = np.arange(len(METHODS))
    for axis, (metric, label) in zip(axes, (("ece", "Delta ECE (points, left is better)"),
                                             ("auarc", "Delta AUARC (points, right is better)"))):
        axis.axvline(0, color="#999999", lw=0.8)
        for offset, estimator in zip((-0.17, 0.17), ESTIMATORS):
            color, marker = ESTIMATOR_STYLE[estimator]
            for position, method in zip(positions, METHODS):
                mean, low, high = bootstrap_mean([100 * row[f"delta_{metric}"]
                                                  for row in select(rows, estimator=estimator, method=method)])
                ablation = family(method) == "Ablation"
                axis.errorbar(mean, position + offset, xerr=[[mean - low], [high - mean]], fmt=marker, color=color,
                              mfc="white" if ablation else color, ms=5, lw=1, capsize=2,
                              label=ESTIMATORS[estimator] if position == 0 else None)
        for boundary in np.cumsum([len(FAMILIES[name]) for name in ("Fixed", "Scalar", "Weighted")]):
            axis.axhline(boundary - 0.5, color="#dddddd", lw=0.8)
        axis.set_xlabel(label)
    axes[0].set_yticks(positions, [LABELS[method] for method in METHODS])
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04))
    figure.suptitle("Pooled confidence in the majority answer vs each group's solo reference\n"
                    "mean over 75 panels (15 groups x 5 datasets), descriptive panel-resampling intervals (overlapping panels); open markers are ablations")
    figure.tight_layout()
    return save(figure, "topline")
