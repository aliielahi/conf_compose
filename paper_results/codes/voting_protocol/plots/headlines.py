"""Topline sentences for the voting scenario, each computed from atomic.csv rather than read off a figure."""

import numpy as np

from accuracy import skill_gain
from common import ESTIMATORS, LABELS, OUT, POOLING, SIZES, TASKS, TASK_NAMES, bootstrap_mean, select
from judges import common_panels, value

RANK_TWINS = [("logodds_sum", "logodds_mean"), ("logodds_sum", "shared_rho"), ("logodds_sum", "shared_scale"),
              ("mean", "blp_equal")]


def mean_delta(rows, estimator, method, metric, **conditions):
    selected = select(rows, estimator=estimator, method=method, **conditions)
    return 100 * np.mean([row[f"delta_{metric}"] for row in selected]) if selected else np.nan


def win_rate(rows, estimator, method, metric):
    selected = select(rows, estimator=estimator, method=method)
    sign = -1 if metric == "ece" else 1
    return np.mean([sign * row[f"delta_{metric}"] > 1e-12 for row in selected])


def twins(rows, first, second):
    """Share of panels in which two methods have the same AUARC."""
    same = []
    for row in select(rows, method=first):
        other = select(rows, method=second, task=row["task"], models=row["models"], estimator=row["estimator"])
        same.append(abs(row["auarc"] - other[0]["auarc"]) < 1e-9)
    return np.mean(same)


def write(rows):
    lines = ["Voting protocol: headline numbers (points = 100 x metric; deltas vs each group's solo reference)",
             "Intervals describe panel resampling; overlapping panels are not independent replications.",
             "Best pooling below compares the eight main methods; ablations are reported separately.", ""]
    for estimator, title in ESTIMATORS.items():
        ranked = sorted(POOLING, key=lambda method: mean_delta(rows, estimator, method, "ece"))
        best = ranked[0]
        ece, low, high = bootstrap_mean([100 * row["delta_ece"] for row in select(rows, estimator=estimator, method=best)])
        auarc = [mean_delta(rows, estimator, method, "auarc") for method in POOLING]
        lines += [f"[{title}]",
                  f"  Best calibrated pooling: {LABELS[best]}, Delta ECE {ece:+.2f} [{low:+.2f}, {high:+.2f}], "
                  f"lower ECE than the solo reference in {win_rate(rows, estimator, best, 'ece'):.0%} of 75 panels.",
                  f"  Runner-up: {LABELS[ranked[1]]} {mean_delta(rows, estimator, ranked[1], 'ece'):+.2f}; "
                  f"worst: {LABELS[ranked[-1]]} {mean_delta(rows, estimator, ranked[-1], 'ece'):+.2f}.",
                  f"  Pooling AUARC deltas range from {min(auarc):+.2f} to {max(auarc):+.2f} "
                  f"(spread between methods {max(auarc) - min(auarc):.2f} points).",
                  f"  Log-odds sum Delta ECE by size: "
                  + ", ".join(f"{size}: {mean_delta(rows, estimator, 'logodds_sum', 'ece', n_models=size):+.1f}" for size in SIZES)
                  + "   (descriptive comparison; model composition also changes with group size)",
                  f"  Shared scale Delta ECE by size: "
                  + ", ".join(f"{size}: {mean_delta(rows, estimator, 'shared_scale', 'ece', n_models=size):+.1f}" for size in SIZES),
                  f"  Weighted BLP Delta ECE by dataset: "
                  + ", ".join(f"{TASK_NAMES[task]} {mean_delta(rows, estimator, 'blp', 'ece', task=task):+.1f}" for task in TASKS),
                  ""]

    lines.append("[Accuracy and AUARC changes]")
    for estimator, title in ESTIMATORS.items():
        selected = select(rows, estimator=estimator, method="mean")
        accuracy = [100 * (row["accuracy"] - row["reference_auarc_accuracy"]) for row in selected]
        delta = [100 * row["delta_auarc"] for row in selected]
        skills = {method: bootstrap_mean([skill_gain(row) for row in select(rows, estimator=estimator, method=method)])[0]
                  for method in ("mean", "logodds_sum", "blp", "logistic_pool")}
        lines.append(f"  {title}: vote accuracy {np.mean(accuracy):+.2f} points over the AUARC reference; "
                     f"corr(accuracy gain, AUARC gain) = {np.corrcoef(accuracy, delta)[0, 1]:.2f}; "
                     "ranking-skill gain " + ", ".join(f"{LABELS[m]} {v:+.1f}" for m, v in skills.items()))
    lines.append("")

    lines.append("[Methods that rank identically: only their calibration differs]")
    for first, second in RANK_TWINS:
        lines.append(f"  {LABELS[first]} and {LABELS[second]}: same AUARC in {twins(rows, first, second):.0%} of 150 panels")
    lines.append("")

    lines.append("[Ablations: unequal weights for BLP; off-diagonal covariance for Kahn]")
    for estimator, title in ESTIMATORS.items():
        for learned, fixed in (("blp", "blp_equal"), ("kahn", "kahn_diagonal")):
            pairs = [(row, select(rows, estimator=estimator, method=fixed, task=row["task"], models=row["models"])[0])
                     for row in select(rows, estimator=estimator, method=learned)]
            ece = np.mean([100 * (a["ece"] - b["ece"]) for a, b in pairs])
            auarc = np.mean([100 * (a["auarc"] - b["auarc"]) for a, b in pairs])
            wins = np.mean([a["ece"] < b["ece"] for a, b in pairs])
            lines.append(f"  {title}: {LABELS[learned]} minus {LABELS[fixed]}: ECE {ece:+.2f}, AUARC {auarc:+.2f}; "
                         f"lower ECE in {wins:.0%} of panels")
        recalibration = np.mean([100 * (b["ece"] - a["ece"]) for a, b in
                                 ((row, select(rows, estimator=estimator, method="blp_equal", task=row["task"],
                                               models=row["models"])[0])
                                  for row in select(rows, estimator=estimator, method="mean"))])
        lines.append(f"  {title}: recalibrating the plain mean (Equal-weight BLP minus Mean, identical ranking): "
                     f"ECE {recalibration:+.2f}")
    lines.append("")

    lines.append("[Judges, on groups where all four judge rows exist, APPROXIMATE reused scores, including changed targets]")
    for estimator, title in ESTIMATORS.items():
        panels = common_panels(rows, estimator)
        for judge, name in (("g3-27i", "27B"), ("l32-3bi", "3B")):
            for metric in ("ece", "auarc"):
                gain = np.mean([value(rows, estimator, f"judge:{judge}:reasoning_confidence", t, m, metric)
                                - value(rows, estimator, f"judge:{judge}:reasoning", t, m, metric) for t, m in panels])
                lines.append(f"  {title} / {name}: showing confidences changes {metric.upper()} by {gain:+.2f}")
        best = np.mean([value(rows, estimator, "blp", t, m, "ece") for t, m in panels])
        judge = np.mean([value(rows, estimator, "judge:g3-27i:reasoning_confidence", t, m, "ece") for t, m in panels])
        lines.append(f"  {title}: on the same {len(panels)} panels, ECE Weighted BLP {best:.2f} vs Judge 27B + conf {judge:.2f}")
    lines.append("")

    lines.append("[Consistency vs sequence probability as the pooled input]")
    for method in ("best_solo", "mean", "blp"):
        pairs = [(row, select(rows, estimator="seq", method=method, task=row["task"], models=row["models"])[0])
                 for row in select(rows, estimator="cons", method=method)]
        ece = np.mean([seq["ece"] < cons["ece"] for cons, seq in pairs])
        auarc = np.mean([seq["auarc"] > cons["auarc"] for cons, seq in pairs])
        lines.append(f"  {LABELS[method]}: sequence has lower ECE in {ece:.0%} and higher AUARC in {auarc:.0%} of panels")

    text = "\n".join(lines) + "\n"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "headlines.txt").write_text(text)
    print(text)
    return [str(OUT / "headlines.txt")]
