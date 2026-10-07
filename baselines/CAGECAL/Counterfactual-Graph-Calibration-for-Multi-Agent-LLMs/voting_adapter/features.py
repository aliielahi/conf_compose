"""Training-only PCA and per-query Pearson matrices (no graph dependencies)."""
from __future__ import annotations

import numpy as np


def answer_projection(embeddings, training_answers, dimension=16):
    """Fit on distinct TRAIN answers; small MC vocabularies are zero-padded."""
    keys = sorted(set(training_answers))
    if not keys:
        raise ValueError("No training answers for PCA")
    x = np.stack([embeddings[k] for k in keys]).astype(np.float64)
    mean = x.mean(axis=0)
    centered = x - mean
    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    rank = min(dimension, max(len(keys) - 1, 0), len(vh))
    basis = vh[:rank]
    projected = {k: np.pad((np.asarray(v) - mean) @ basis.T, (0, dimension - rank)).astype(np.float32)
                 for k, v in embeddings.items()}
    return projected, dict(mean=mean, basis=basis, singular=singular, training_answers=np.asarray(keys))


def pearson_matrix(correctness, n_agents):
    from cage_cal.per_query_W import _pearson_W
    x = np.asarray(correctness, dtype=np.float32).reshape(-1, n_agents)
    return _pearson_W(x)


def query_matrices(rows, question_embeddings, k=20):
    """Neighbors from this panel's training questions, excluding query itself.

    Validation/evaluation labels are never read by this function. Every W row
    records its training neighbor IDs so the supervision can be audited.
    """
    groups = {}
    for row in rows:
        groups.setdefault((row["task"], row["models"]), []).append(row)
    out = {}
    for (task, models), group in groups.items():
        train = sorted((r for r in group if r["split"] == "train"), key=lambda r: r["id"])
        if len(train) < 2:
            raise ValueError(f"{task}/{models}: fewer than two usable training questions for W")
        ref = np.stack([question_embeddings[(task, r["id"])] for r in train])
        labels = np.asarray([r["member_correct"] for r in train], dtype=np.float32)
        for row in group:
            similarity = ref @ question_embeddings[(task, row["id"])]
            order = np.argsort(-similarity, kind="stable")
            indices = [int(i) for i in order if train[i]["id"] != row["id"]][:k]
            out[(task, models, row["id"])] = (
                pearson_matrix(labels[indices], len(row["model_ids"])),
                [train[i]["id"] for i in indices])
    return out
