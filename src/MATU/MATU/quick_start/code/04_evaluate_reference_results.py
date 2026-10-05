"""Evaluate included quick-start reference results.

By default this evaluates the MATH + Qwen2.5 quick-start artifact. Use
``--sample mmlu-autogen-qwen`` to evaluate the included MMLU + AutoGen +
Qwen2.5 artifact.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import auc, roc_curve

from env_utils import load_quickstart_env


QUICK_START = Path(__file__).resolve().parents[1]
REPO_ROOT = QUICK_START.parent
PAPER_AUROC_EXPANSION = 20

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from matu.evaluate_uncertainty import accuracy_from_labels, auarc, auroc_binary  # noqa: E402
from matu.fit_to_uncertainty import uncertainty_from_matu_result  # noqa: E402


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def save_pickle(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=4)


def evaluate_math_qwen() -> None:
    fit_path = QUICK_START / "results" / "fit_dict_Math_qwen2.5_qwen3embedding.pkl"
    labels_path = QUICK_START / "results" / "accuracy_dict_Math_qwen2.5.pkl"
    uncertainty_path = QUICK_START / "results" / "uncertainty_Math_qwen2.5.pkl"

    if not uncertainty_path.exists():
        fit_dict = load_pickle(fit_path)
        uncertainty = {key: uncertainty_from_matu_result(value) for key, value in fit_dict.items()}
        save_pickle(uncertainty, uncertainty_path)

    uncertainty_dict = load_pickle(uncertainty_path)
    labels = load_pickle(labels_path)
    keys = sorted(set(uncertainty_dict) & set(labels))
    if not keys:
        raise ValueError("No overlapping task keys between uncertainty and labels.")

    scores = np.asarray([float(uncertainty_dict[key]) for key in keys], dtype=float)
    accuracies = np.asarray([accuracy_from_labels(labels[key]) for key in keys], dtype=float)
    y_error = (accuracies < 1.0).astype(int)

    print("MATH + Qwen2.5-7B")
    print(f"Tasks: {len(keys)}")
    print(f"Mean accuracy: {accuracies.mean():.4f}")
    print(f"AUROC: {auroc_binary(y_error, scores):.4f}")
    print(f"AUARC: {auarc(accuracies, scores):.4f}")


def label_accuracy(labels: Any) -> float:
    values = list(labels)
    if not values:
        return 0.0
    correct = sum(1 for item in values if str(item).strip().lower() == "correct")
    return correct / len(values)


def fit_sum(fit_values: Any) -> float:
    return float(np.nansum(np.asarray(fit_values, dtype=float)))


def paper_auroc(accuracies: np.ndarray, certainty_scores: np.ndarray) -> float:
    y_true: list[int] = []
    y_score: list[float] = []
    for accuracy, score in zip(accuracies, certainty_scores):
        correct_runs = int(round(float(accuracy) * PAPER_AUROC_EXPANSION))
        for idx in range(PAPER_AUROC_EXPANSION):
            y_true.append(1 if idx < correct_runs else 0)
            y_score.append(float(score))
    fpr, tpr, _ = roc_curve(y_true, y_score, drop_intermediate=False)
    return float(auc(fpr, tpr))


def evaluate_mmlu_autogen_qwen() -> None:
    fit_path = QUICK_START / "results" / "fit_dict_MMLU_Autogen_qwen2.5.pkl"
    labels_path = QUICK_START / "results" / "accuracy_dict_MMLU_Autogen_qwen2.5.pkl"
    fit_dict = load_pickle(fit_path)
    labels = load_pickle(labels_path)

    keys = sorted(set(fit_dict) & set(labels))
    if not keys:
        raise ValueError("No overlapping task keys between fit_dict and labels.")

    certainty_scores = np.asarray([fit_sum(fit_dict[key]) for key in keys], dtype=float)
    rank_counts = np.asarray([len(fit_dict[key]) for key in keys], dtype=float)
    uncertainty_scores = rank_counts - certainty_scores
    accuracies = np.asarray([label_accuracy(labels[key]) for key in keys], dtype=float)

    print("MMLU + AutoGen + Qwen2.5-7B")
    print(f"Tasks: {len(keys)}")
    print(f"Mean accuracy: {accuracies.mean():.4f}")
    print(f"AUROC: {paper_auroc(accuracies, certainty_scores):.4f}")
    print(f"AUARC: {auarc(accuracies, uncertainty_scores):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate included MATU quick-start reference results.")
    parser.add_argument(
        "--sample",
        choices=["math-qwen", "mmlu-autogen-qwen", "all"],
        default="math-qwen",
        help="Reference sample to evaluate.",
    )
    args = parser.parse_args()

    load_quickstart_env()

    if args.sample in {"math-qwen", "all"}:
        evaluate_math_qwen()
    if args.sample == "all":
        print()
    if args.sample in {"mmlu-autogen-qwen", "all"}:
        evaluate_mmlu_autogen_qwen()


if __name__ == "__main__":
    main()
