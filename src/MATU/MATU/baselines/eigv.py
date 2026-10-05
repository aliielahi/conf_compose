"""EigV-style agreement baseline for multi-run conversation logs.

The original experiments compare against an entailment-graph baseline. This
public script provides a self-contained implementation:

1. Convert each run into a text sample, using either the final assistant answer
   or the whole conversation.
2. Use an NLI model to estimate pairwise agreement.
3. Build a disagreement graph and score uncertainty with graph eigenvalues.

Higher output scores indicate higher uncertainty.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


LOGGER = logging.getLogger(__name__)


def load_logs(path: Path) -> dict[str, list[list[dict[str, Any]]]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def final_assistant_text(run: list[dict[str, Any]]) -> str:
    last = ""
    for turn in run:
        if turn.get("role") == "assistant":
            last = str(turn.get("output", turn.get("content", "")))
    return last


def whole_conversation_text(run: list[dict[str, Any]]) -> str:
    lines = []
    for turn in run:
        role = str(turn.get("role", "unknown"))
        text = str(turn.get("output", turn.get("content", "")))
        lines.append(f"{role}: {text}")
    return "\n".join(lines)


class NLIAgreement:
    def __init__(self, model_name: str, device: str | None = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        self.entail_idx = next((i for i, label in id2label.items() if "entail" in label), None)
        self.contra_idx = next((i for i, label in id2label.items() if "contrad" in label), None)
        if self.entail_idx is None:
            self.entail_idx = int(np.argmax([1 if "entail" in v else 0 for v in id2label.values()]))

    @torch.inference_mode()
    def entailment_probability(self, premise: str, hypothesis: str) -> float:
        inputs = self.tokenizer(
            premise,
            hypothesis,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)
        logits = self.model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1)
        return float(probs[self.entail_idx].detach().cpu())

    def symmetric_agreement(self, a: str, b: str) -> float:
        if not a.strip() or not b.strip():
            return 0.0
        return 0.5 * (self.entailment_probability(a, b) + self.entailment_probability(b, a))


def eigv_uncertainty(texts: list[str], nli: NLIAgreement, score: str) -> dict[str, Any]:
    n = len(texts)
    agreement = np.eye(n, dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            value = nli.symmetric_agreement(texts[i], texts[j])
            agreement[i, j] = value
            agreement[j, i] = value

    disagreement = 1.0 - agreement
    np.fill_diagonal(disagreement, 0.0)
    degrees = disagreement.sum(axis=1)
    laplacian = np.diag(degrees) - disagreement
    eigvals = np.linalg.eigvalsh(laplacian)
    eigvals = np.maximum(eigvals, 0.0)

    if score == "sum":
        uncertainty = float(np.sum(eigvals))
    elif score == "spectral_entropy":
        total = float(np.sum(eigvals))
        probs = eigvals / total if total > 0 else np.zeros_like(eigvals)
        probs = probs[probs > 0]
        uncertainty = float(-np.sum(probs * np.log(probs))) if probs.size else 0.0
    elif score == "lambda_max":
        uncertainty = float(np.max(eigvals)) if eigvals.size else 0.0
    else:
        raise ValueError(score)

    return {
        "uncertainty": uncertainty,
        "eigenvalues": eigvals.tolist(),
        "agreement": agreement.tolist(),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    parser = argparse.ArgumentParser(description="Compute an EigV-style NLI graph uncertainty baseline.")
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=["final", "whole"], default="final")
    parser.add_argument("--model", default="roberta-large-mnli")
    parser.add_argument("--score", choices=["sum", "spectral_entropy", "lambda_max"], default="sum")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    logs = load_logs(args.logs)
    nli = NLIAgreement(args.model, args.device)
    extract = final_assistant_text if args.mode == "final" else whole_conversation_text

    results = {}
    for key, runs in tqdm(logs.items(), desc="EigV"):
        texts = [extract(run) for run in runs]
        results[key] = eigv_uncertainty(texts, nli, args.score)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as f:
        pickle.dump(results, f, protocol=4)
    LOGGER.info("Saved EigV scores to %s", args.out)


if __name__ == "__main__":
    main()
