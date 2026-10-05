"""Evaluate uncertainty scores against correctness labels.

Supports both the new MATU output format from ``cp2_matu.py`` and the legacy
``fit_dict`` format used in the original experiment scripts.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def score_from_value(value: Any, mode: str) -> float:
    if mode == "auto":
        if isinstance(value, dict):
            if any(k in value for k in ("uncertainty", "relative_loss", "fit")):
                mode = "legacy_fit"
            else:
                mode = "raw"
        else:
            arr = np.asarray(value, dtype=float)
            mode = "raw" if arr.ndim == 0 else "legacy_fit"

    if isinstance(value, dict):
        if "uncertainty" in value:
            return float(value["uncertainty"])
        if "relative_loss" in value:
            return float(np.nansum(value["relative_loss"]))
        if "fit" in value:
            fits = np.asarray(value["fit"], dtype=float)
            return float(np.nansum(1.0 - fits))

    arr = np.asarray(value, dtype=float)
    if mode == "legacy_fit":
        return float(np.nansum(1.0 - arr))
    if mode == "raw":
        return float(np.nanmean(arr))
    raise ValueError(f"Unsupported uncertainty value for mode={mode}: {type(value)}")


def accuracy_from_labels(labels: Any) -> float:
    if isinstance(labels, (str, bytes)):
        values = [labels]
    else:
        values = list(labels)

    if not values:
        return 0.0

    correct = 0
    for item in values:
        text = str(item).strip().lower()
        if text in {"correct", "1", "true"}:
            correct += 1
        elif text in {"incorrect", "0", "false"}:
            correct += 0
        else:
            correct += 1 if "correct" in text and "incorrect" not in text else 0
    return correct / len(values)


def auroc_binary(y_error: np.ndarray, uncertainty: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y_error, uncertainty))
    except Exception:
        pos = uncertainty[y_error == 1]
        neg = uncertainty[y_error == 0]
        if len(pos) == 0 or len(neg) == 0:
            return float("nan")
        wins = 0.0
        total = 0.0
        for p in pos:
            wins += float(np.sum(p > neg)) + 0.5 * float(np.sum(p == neg))
            total += len(neg)
        return wins / total


def auarc(accuracy: np.ndarray, uncertainty: np.ndarray) -> float:
    """Area under accuracy-rejection curve.

    Low-uncertainty examples are retained first; as coverage increases, we
    average the retained accuracies.
    """
    order = np.argsort(uncertainty)
    sorted_acc = accuracy[order]
    coverages = np.arange(1, len(sorted_acc) + 1, dtype=float) / len(sorted_acc)
    acc_curve = np.cumsum(sorted_acc) / np.arange(1, len(sorted_acc) + 1)
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(acc_curve, coverages))
    return float(np.trapz(acc_curve, coverages))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate uncertainty scores with AUROC and AUARC.")
    parser.add_argument("--uncertainty", type=Path, required=True, help="MATU output or legacy fit_dict pickle.")
    parser.add_argument("--labels", type=Path, required=True, help="accuracy_dict pickle.")
    parser.add_argument("--score_mode", choices=["auto", "legacy_fit", "raw"], default="auto")
    parser.add_argument(
        "--error_rule",
        choices=["any_incorrect", "majority_incorrect"],
        default="any_incorrect",
        help="How to binarize repeated-run correctness for AUROC.",
    )
    args = parser.parse_args()

    uncertainty_dict = load_pickle(args.uncertainty)
    labels_dict = load_pickle(args.labels)

    keys = sorted(set(uncertainty_dict) & set(labels_dict))
    if not keys:
        raise ValueError("No overlapping keys between uncertainty and labels.")

    mode = args.score_mode
    scores = np.asarray([score_from_value(uncertainty_dict[k], mode) for k in keys], dtype=float)
    accuracies = np.asarray([accuracy_from_labels(labels_dict[k]) for k in keys], dtype=float)

    if args.error_rule == "any_incorrect":
        y_error = (accuracies < 1.0).astype(int)
    else:
        y_error = (accuracies < 0.5).astype(int)

    print(f"Tasks: {len(keys)}")
    print(f"Mean accuracy: {accuracies.mean():.4f}")
    print(f"AUROC: {auroc_binary(y_error, scores):.4f}")
    print(f"AUARC: {auarc(accuracies, scores):.4f}")


if __name__ == "__main__":
    main()
