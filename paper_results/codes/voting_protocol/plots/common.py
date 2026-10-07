"""Shared loading, method naming and figure helpers for the voting-protocol plots."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
RESULTS = ROOT / "paper_results/results/voting_protocol"
OUT = RESULTS / "plots"

TASKS = ("csqa", "boolq", "gsm8k", "truthfulqa", "gpqa")
TASK_NAMES = {"csqa": "CSQA", "boolq": "BoolQ", "gsm8k": "GSM8K", "truthfulqa": "TruthfulQA", "gpqa": "GPQA"}
ESTIMATORS = {"cons": "Consistency", "seq": "Sequence probability"}
SIZES = (2, 3, 4, 5, 6)

# Families: no fitting, scalar recalibration of the logit sum, learned per-member weights, ablations, judges.
FAMILIES = {
    "Fixed": ["mean", "logodds_sum", "logodds_mean"],
    "Scalar": ["shared_rho", "shared_scale"],
    "Weighted": ["kahn", "blp", "logistic_pool"],
    "Ablation": ["blp_equal", "kahn_diagonal"],
    "Judge": ["judge:g3-27i:reasoning", "judge:g3-27i:reasoning_confidence",
              "judge:l32-3bi:reasoning", "judge:l32-3bi:reasoning_confidence"],
}
POOLING = FAMILIES["Fixed"] + FAMILIES["Scalar"] + FAMILIES["Weighted"]
LABELS = {
    "best_solo": "Solo reference", "mean": "Mean", "logodds_sum": "Log-odds sum", "logodds_mean": "Log-odds mean",
    "shared_rho": "Shared rho", "shared_scale": "Shared scale", "kahn": "Kahn", "blp": "Weighted BLP",
    "logistic_pool": "Logistic pool", "blp_equal": "Equal-weight BLP", "kahn_diagonal": "Kahn (diagonal)",
    "judge:g3-27i:reasoning": "Judge 27B", "judge:g3-27i:reasoning_confidence": "Judge 27B + conf",
    "judge:l32-3bi:reasoning": "Judge 3B", "judge:l32-3bi:reasoning_confidence": "Judge 3B + conf",
}
COLORS = {
    "best_solo": "#555555", "mean": "#9ecae1", "logodds_sum": "#3182bd", "logodds_mean": "#08519c",
    "shared_rho": "#a1d99b", "shared_scale": "#238b45", "kahn": "#fd8d3c", "blp": "#d94801",
    "logistic_pool": "#7f2704", "blp_equal": "#fdae6b", "kahn_diagonal": "#fdd0a2",
    "judge:g3-27i:reasoning": "#bcbddc", "judge:g3-27i:reasoning_confidence": "#6a51a3",
    "judge:l32-3bi:reasoning": "#dadaeb", "judge:l32-3bi:reasoning_confidence": "#9e9ac8",
}
MARKERS = {family: marker for family, marker in zip(FAMILIES, "osD^v")}
NUMERIC = ("n_models", "n_matched", "coverage", "accuracy", "ece", "reference_ece", "delta_ece", "auarc",
           "reference_auarc", "delta_auarc", "reference_ece_accuracy", "reference_auarc_accuracy")


def family(method):
    return next((name for name, members in FAMILIES.items() if method in members), "Reference")


def load():
    """atomic.csv rows with numbers parsed, plus the audit keyed by (task, estimator, models)."""
    rows = []
    with (RESULTS / "atomic.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            for key in NUMERIC:
                row[key] = float(row[key]) if row[key] not in ("", None) else None
            row["n_models"] = int(row["n_models"])
            rows.append(row)
    approximate = any(row.get("judge_approximate") == "True" for row in rows)
    for method in FAMILIES["Judge"]:
        if approximate and "approx." not in LABELS[method]:
            LABELS[method] += " (approx.)"
    audit = {(cell["task"], cell["estimator"], cell["models"]): cell
             for cell in json.loads((RESULTS / "audit.json").read_text())}
    return rows, audit


def select(rows, **conditions):
    return [row for row in rows if all(row[key] == value for key, value in conditions.items())]


def oracle_auarc(accuracy, n):
    """AUARC of a perfect ranking: every correct answer first, then selective accuracy decays as m/k."""
    m = int(round(accuracy * n))
    if m == 0:
        return 0.0
    harmonic = np.cumsum(1 / np.arange(1, n + 1))
    return (m + m * (harmonic[n - 1] - harmonic[m - 1])) / n


def ranking_skill(auarc, accuracy, n):
    """0 for a random ranking (AUARC = accuracy), 1 for the oracle: AUARC with the accuracy effect removed."""
    ceiling = oracle_auarc(accuracy, int(n)) - accuracy
    return (auarc - accuracy) / ceiling if ceiling > 1e-9 else np.nan


def bootstrap_mean(values, draws=2000, seed=0):
    """Mean and a descriptive 95% interval over panels; panels share models, so this is not a test."""
    values = np.asarray([value for value in values if value is not None and np.isfinite(value)], dtype=float)
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(draws, len(values))).mean(axis=1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def style():
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
                         "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
                         "savefig.bbox": "tight", "pdf.fonttype": 42})


def save(figure, name):
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = RESULTS / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()).get("judge_policy") == "approximate":
        figure.text(0.5, -0.025, "Confidence tie-break; fitting-accuracy solo reference. Judge scores reused approximately, including changed targets.",
                    ha="center", va="top", fontsize=7)
    for suffix in ("pdf", "png"):
        figure.savefig(OUT / f"{name}.{suffix}")
    plt.close(figure)
    return [str(OUT / f"{name}.{suffix}") for suffix in ("pdf", "png")]
