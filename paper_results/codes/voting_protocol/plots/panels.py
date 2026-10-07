import argparse

import matplotlib.pyplot as plt
import numpy as np

from common import ESTIMATORS, FAMILIES, LABELS, TASKS, TASK_NAMES, load, save, select, style

METHODS = [method for family in FAMILIES.values() for method in family]


def matrix(rows, estimator, task, metric, groups, methods):
    values = np.full((len(groups), len(methods)), np.nan)
    for i, (_, models) in enumerate(groups):
        for j, method in enumerate(methods):
            matches = select(rows, estimator=estimator, task=task, models=models, method=method)
            if len(matches) > 1:
                raise ValueError(f"duplicate atomic result: {estimator}/{task}/{models}/{method}")
            if matches:
                values[i, j] = 100 * matches[0][f"delta_{metric}"]
    return values


def draw(rows, tasks=TASKS, estimators=tuple(ESTIMATORS), methods=METHODS):
    ordered = sorted({(row["n_models"], row["models"]) for row in rows})
    paths = []
    for estimator in estimators:
        for task in tasks:
            if not select(rows, estimator=estimator, task=task):
                continue
            figure, axes = plt.subplots(1, 2, figsize=(20, 12), sharey=True)
            for axis, metric in zip(axes, ("ece", "auarc")):
                values = matrix(rows, estimator, task, metric, ordered, methods)
                finite = values[np.isfinite(values)]
                bound = max(1.0, float(np.max(np.abs(finite)))) if len(finite) else 1.0
                cmap = plt.get_cmap("RdBu_r" if metric == "ece" else "RdBu").copy()
                cmap.set_bad("#dddddd")
                image = axis.imshow(values, cmap=cmap, vmin=-bound, vmax=bound, aspect="auto")
                for i in range(len(ordered)):
                    for j in range(len(methods)):
                        value = values[i, j]
                        label = f"{value:+.1f}" if np.isfinite(value) else "--"
                        color = "white" if np.isfinite(value) and abs(value) > 0.65 * bound else "black"
                        axis.text(j, i, label, ha="center", va="center", fontsize=7, color=color)
                for i in range(1, len(ordered)):
                    if ordered[i][0] != ordered[i - 1][0]:
                        axis.axhline(i - 0.5, color="black", lw=1)
                axis.set_xticks(range(len(methods)), [LABELS[method] for method in methods], rotation=60, ha="right", fontsize=8)
                axis.set_yticks(range(len(ordered)), [f"G{i + 1:02d} ({size} voters)" for i, (size, _) in enumerate(ordered)])
                axis.set_title(f"Delta {metric.upper()} vs solo reference ({'negative' if metric == 'ece' else 'positive'} is better)")
                figure.colorbar(image, ax=axis, fraction=0.025, pad=0.02, label="Percentage points")
            figure.suptitle(f"{TASK_NAMES[task]} / {ESTIMATORS[estimator]}: every group and method", y=0.96, fontsize=15)
            figure.text(0.5, 0.92, "Blue = improvement; red = deterioration; gray = missing. These are effect sizes, not significance tests.", ha="center", fontsize=10)
            figure.subplots_adjust(left=0.075, right=0.96, top=0.87, bottom=0.35, wspace=0.23)
            midpoint = (len(ordered) + 1) // 2
            for i, (size, models) in enumerate(ordered):
                column, line = divmod(i, midpoint)
                figure.text(0.04 + column * 0.49, 0.19 - line * 0.02,
                            f"G{i + 1:02d}: " + models.replace("|", ", "), fontsize=8)
            paths += save(figure, f"panels_{task}_{estimator}")
    return paths


def main():
    parser = argparse.ArgumentParser(description="Every voting group by every confidence-combination method")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--estimators", nargs="+", choices=list(ESTIMATORS), default=list(ESTIMATORS))
    args = parser.parse_args()
    style()
    rows, _ = load()
    for path in draw(rows, args.tasks, args.estimators):
        print(path)


if __name__ == "__main__":
    main()
