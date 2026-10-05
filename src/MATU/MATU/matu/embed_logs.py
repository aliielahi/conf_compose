"""Embed multi-agent conversation logs into per-role trajectory matrices.

Expected log format:
{
  "task_id": [
    [
      {"role": "user", "output": "..."},
      {"role": "assistant", "output": "..."}
    ],
    ...
  ]
}

The output is one pickle per requested role:
  <role>_embedding_matrices.pkl

Each pickle stores:
  dict[str, list[np.ndarray]]
where each key maps to a list of runs, and each run is a matrix with shape
(num_steps_for_role, embedding_dim).
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from matu.constants import DEFAULT_EMBEDDING_MODEL


LOGGER = logging.getLogger(__name__)


def load_logs(path: Path) -> dict[str, list[list[dict[str, Any]]]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def encode_texts(model: SentenceTransformer, texts: list[str], dim: int) -> np.ndarray:
    if not texts:
        return np.empty((0, dim), dtype=np.float32)
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(emb, dtype=np.float32)


def embed_logs(
    logs: dict[str, list[list[dict[str, Any]]]],
    model: SentenceTransformer,
    roles: list[str],
) -> dict[str, dict[str, list[np.ndarray]]]:
    dim = int(model.get_sentence_embedding_dimension() or 0)
    if dim <= 0:
        probe = model.encode(["dimension probe"], normalize_embeddings=True)
        dim = int(np.asarray(probe).shape[-1])

    outputs: dict[str, dict[str, list[np.ndarray]]] = {role: {} for role in roles}
    role_set = set(roles)

    for key, runs in tqdm(logs.items(), desc="Embedding tasks"):
        for role in roles:
            outputs[role][key] = []

        for run in runs:
            texts_by_role: dict[str, list[str]] = {role: [] for role in roles}
            for turn in run:
                role = str(turn.get("role", ""))
                if role not in role_set:
                    continue
                text = turn.get("output", turn.get("content", ""))
                texts_by_role[role].append(text if isinstance(text, str) else str(text))

            for role in roles:
                outputs[role][key].append(encode_texts(model, texts_by_role[role], dim))

    return outputs


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    parser = argparse.ArgumentParser(description="Create embedding matrices from MAS conversation logs.")
    parser.add_argument("--logs", type=Path, required=True, help="Path to conversation log JSON.")
    parser.add_argument("--out_dir", type=Path, required=True, help="Directory for output pickle files.")
    parser.add_argument("--roles", nargs="+", default=["user", "assistant"], help="Roles to embed.")
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL, help="SentenceTransformer embedding model.")
    parser.add_argument("--cache_dir", default=None, help="Optional Hugging Face cache directory.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    logs = load_logs(args.logs)
    model = SentenceTransformer(
        args.model,
        cache_folder=args.cache_dir,
        model_kwargs={"device_map": "auto"},
        tokenizer_kwargs={"padding_side": "left"},
    )

    outputs = embed_logs(logs, model, args.roles)
    for role, data in outputs.items():
        out_path = args.out_dir / f"{role}_embedding_matrices.pkl"
        with out_path.open("wb") as f:
            pickle.dump(data, f, protocol=4)
        LOGGER.info("Saved %s embeddings to %s", role, out_path)

    metadata = {
        "embedding_model": args.model,
        "embedding_dim": int(model.get_sentence_embedding_dimension() or 0),
        "roles": args.roles,
    }
    metadata_path = args.out_dir / "embedding_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    LOGGER.info("Saved embedding metadata to %s", metadata_path)


if __name__ == "__main__":
    main()
