"""Unified command-line interface for MATU."""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from matu.config import choose, load_config, section
from matu.constants import DEFAULT_EMBEDDING_MODEL


LOGGER = logging.getLogger("matu")


def _path(value: Any) -> Path | None:
    if value is None:
        return None
    return value if isinstance(value, Path) else Path(value)


def _path_list(value: Any) -> list[Path]:
    if value is None:
        return []
    return [item if isinstance(item, Path) else Path(item) for item in value]


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )


def run_embed(args: argparse.Namespace) -> None:
    from sentence_transformers import SentenceTransformer

    from matu.embed_logs import embed_logs, load_logs

    cfg = section(load_config(args.config), "embedding")
    logs_path = _path(choose(args.logs, cfg, "logs"))
    out_dir = _path(choose(args.out_dir, cfg, "out_dir"))
    roles = choose(args.roles, cfg, "roles", ["user", "assistant"])
    model_name = choose(args.model, cfg, "model", DEFAULT_EMBEDDING_MODEL)
    cache_dir = choose(args.cache_dir, cfg, "cache_dir")

    if logs_path is None or out_dir is None:
        raise SystemExit("embed requires --logs and --out_dir, or embedding.logs/out_dir in the config.")

    out_dir.mkdir(parents=True, exist_ok=True)
    logs = load_logs(logs_path)
    LOGGER.info("Loading embedding model: %s", model_name)
    model = SentenceTransformer(
        model_name,
        cache_folder=cache_dir,
        model_kwargs={"device_map": "auto"},
        tokenizer_kwargs={"padding_side": "left"},
    )

    outputs = embed_logs(logs, model, list(roles))
    for role, data in outputs.items():
        out_path = out_dir / f"{role}_embedding_matrices.pkl"
        with out_path.open("wb") as f:
            pickle.dump(data, f, protocol=4)
        LOGGER.info("Saved %s embeddings to %s", role, out_path)

    metadata = {
        "embedding_model": model_name,
        "embedding_dim": int(model.get_sentence_embedding_dimension() or 0),
        "roles": list(roles),
    }
    metadata_path = out_dir / "embedding_metadata.json"
    import json

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    LOGGER.info("Saved embedding metadata to %s", metadata_path)


def run_cp2(args: argparse.Namespace) -> None:
    from matu.cp2_matu import build_matrix_list, load_embedding_pickle, parafac2_als

    cfg = section(load_config(args.config), "cp2")
    embeddings = _path_list(choose(args.embeddings, cfg, "embeddings"))
    out_path = _path(choose(args.out, cfg, "out"))
    legacy_fit_out = _path(choose(args.legacy_fit_out, cfg, "legacy_fit_out"))
    min_rank = int(choose(args.min_rank, cfg, "min_rank", 1))
    max_rank = int(choose(args.max_rank, cfg, "max_rank", 50))
    max_iter = int(choose(args.max_iter, cfg, "max_iter", 25))
    tol = float(choose(args.tol, cfg, "tol", 1e-6))
    seed = int(choose(args.seed, cfg, "seed", 0))
    combine_mode = choose(args.combine_mode, cfg, "combine_mode", "interleave")
    normalize = bool(choose(args.normalize, cfg, "normalize", True))

    if not embeddings or out_path is None:
        raise SystemExit("cp2 requires --embeddings and --out, or cp2.embeddings/out in the config.")

    embedding_dicts = [load_embedding_pickle(path) for path in embeddings]
    keys = sorted(set.intersection(*(set(d.keys()) for d in embedding_dicts)))
    if not keys:
        raise ValueError("No overlapping task keys across embedding pickles.")

    ranks = list(range(min_rank, max_rank + 1))
    results: dict[str, dict[str, Any]] = {}
    legacy_fit: dict[str, list[float]] = {}

    for key in tqdm(keys, desc="MATU"):
        matrices = build_matrix_list(
            embedding_dicts,
            key,
            combine_mode=combine_mode,
            normalize=normalize,
        )
        fit_values = []
        losses = []
        relative_losses = []
        for rank in ranks:
            try:
                _, fit, loss, rel_loss = parafac2_als(
                    matrices,
                    rank,
                    max_iter=max_iter,
                    tol=tol,
                    seed=seed,
                )
            except Exception as exc:
                LOGGER.warning("key=%s rank=%s failed: %s", key, rank, exc)
                fit, loss, rel_loss = 0.0, float("nan"), 1.0
            fit_values.append(float(fit))
            losses.append(float(loss))
            relative_losses.append(float(rel_loss))

        results[key] = {
            "ranks": ranks,
            "fit": fit_values,
            "loss": losses,
            "relative_loss": relative_losses,
            "uncertainty": float(np.nansum(relative_losses)),
        }
        legacy_fit[key] = fit_values

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(results, f, protocol=4)
    LOGGER.info("Saved MATU scores to %s", out_path)

    if legacy_fit_out is not None:
        legacy_fit_out.parent.mkdir(parents=True, exist_ok=True)
        with legacy_fit_out.open("wb") as f:
            pickle.dump(legacy_fit, f, protocol=4)
        LOGGER.info("Saved legacy fit_dict to %s", legacy_fit_out)


def run_fit(args: argparse.Namespace) -> None:
    from matu.fit_to_uncertainty import load_pickle, save_pickle, uncertainty_from_matu_result

    cfg = section(load_config(args.config), "fit")
    fit_dict_path = _path(choose(args.fit_dict, cfg, "fit_dict"))
    out_path = _path(choose(args.out, cfg, "out"))
    if fit_dict_path is None or out_path is None:
        raise SystemExit("fit requires --fit_dict and --out, or fit.fit_dict/out in the config.")

    fit_dict = load_pickle(fit_dict_path)
    if not isinstance(fit_dict, dict):
        raise TypeError("--fit_dict must contain a dict")
    uncertainty = {key: uncertainty_from_matu_result(value) for key, value in fit_dict.items()}
    save_pickle(uncertainty, out_path)
    LOGGER.info("Saved uncertainty dict to %s", out_path)


def run_eval(args: argparse.Namespace) -> None:
    from matu.evaluate_uncertainty import accuracy_from_labels, auarc, auroc_binary, load_pickle, score_from_value

    cfg = section(load_config(args.config), "evaluation")
    uncertainty_path = _path(choose(args.uncertainty, cfg, "uncertainty"))
    labels_path = _path(choose(args.labels, cfg, "labels"))
    score_mode = choose(args.score_mode, cfg, "score_mode", "auto")
    error_rule = choose(args.error_rule, cfg, "error_rule", "any_incorrect")
    if uncertainty_path is None or labels_path is None:
        raise SystemExit("eval requires --uncertainty and --labels, or evaluation.uncertainty/labels in the config.")

    uncertainty_dict = load_pickle(uncertainty_path)
    labels_dict = load_pickle(labels_path)
    keys = sorted(set(uncertainty_dict) & set(labels_dict))
    if not keys:
        raise ValueError("No overlapping keys between uncertainty and labels.")

    scores = np.asarray([score_from_value(uncertainty_dict[k], score_mode) for k in keys], dtype=float)
    accuracies = np.asarray([accuracy_from_labels(labels_dict[k]) for k in keys], dtype=float)
    if error_rule == "any_incorrect":
        y_error = (accuracies < 1.0).astype(int)
    else:
        y_error = (accuracies < 0.5).astype(int)

    print(f"Tasks: {len(keys)}")
    print(f"Mean accuracy: {accuracies.mean():.4f}")
    print(f"AUROC: {auroc_binary(y_error, scores):.4f}")
    print(f"AUARC: {auarc(accuracies, scores):.4f}")


def run_eigv(args: argparse.Namespace) -> None:
    from baselines.eigv import NLIAgreement, eigv_uncertainty, final_assistant_text, load_logs, whole_conversation_text

    cfg = section(load_config(args.config), "eigv")
    logs_path = _path(choose(args.logs, cfg, "logs"))
    out_path = _path(choose(args.out, cfg, "out"))
    mode = choose(args.mode, cfg, "mode", "final")
    model_name = choose(args.model, cfg, "model", "roberta-large-mnli")
    score = choose(args.score, cfg, "score", "sum")
    device = choose(args.device, cfg, "device")
    if logs_path is None or out_path is None:
        raise SystemExit("eigv requires --logs and --out, or eigv.logs/out in the config.")

    logs = load_logs(logs_path)
    nli = NLIAgreement(model_name, device)
    extract = final_assistant_text if mode == "final" else whole_conversation_text

    results = {}
    for key, runs in tqdm(logs.items(), desc="EigV"):
        texts = [extract(run) for run in runs]
        results[key] = eigv_uncertainty(texts, nli, score)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(results, f, protocol=4)
    LOGGER.info("Saved EigV scores to %s", out_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MATU: Multi-Agent Tensor Uncertainty")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    embed = subparsers.add_parser("embed", help="Embed conversation logs into role trajectory matrices.")
    embed.add_argument("--config", type=Path, default=None)
    embed.add_argument("--logs", type=Path, default=None)
    embed.add_argument("--out_dir", type=Path, default=None)
    embed.add_argument("--roles", nargs="+", default=None)
    embed.add_argument("--model", default=None)
    embed.add_argument("--cache_dir", default=None)
    embed.set_defaults(func=run_embed)

    cp2 = subparsers.add_parser("cp2", help="Run CP-2/PARAFAC2 MATU scoring.")
    cp2.add_argument("--config", type=Path, default=None)
    cp2.add_argument("--embeddings", type=Path, nargs="+", default=None)
    cp2.add_argument("--out", type=Path, default=None)
    cp2.add_argument("--legacy_fit_out", type=Path, default=None)
    cp2.add_argument("--min_rank", type=int, default=None)
    cp2.add_argument("--max_rank", type=int, default=None)
    cp2.add_argument("--max_iter", type=int, default=None)
    cp2.add_argument("--tol", type=float, default=None)
    cp2.add_argument("--seed", type=int, default=None)
    cp2.add_argument("--combine_mode", choices=["interleave", "concat_steps"], default=None)
    cp2.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=None)
    cp2.set_defaults(func=run_cp2)

    fit = subparsers.add_parser("fit", help="Convert fit curves/MATU output to scalar uncertainty.")
    fit.add_argument("--config", type=Path, default=None)
    fit.add_argument("--fit_dict", type=Path, default=None)
    fit.add_argument("--out", type=Path, default=None)
    fit.set_defaults(func=run_fit)

    evaluate = subparsers.add_parser("eval", help="Evaluate uncertainty with AUROC and AUARC.")
    evaluate.add_argument("--config", type=Path, default=None)
    evaluate.add_argument("--uncertainty", type=Path, default=None)
    evaluate.add_argument("--labels", type=Path, default=None)
    evaluate.add_argument("--score_mode", choices=["auto", "legacy_fit", "raw"], default=None)
    evaluate.add_argument("--error_rule", choices=["any_incorrect", "majority_incorrect"], default=None)
    evaluate.set_defaults(func=run_eval)

    eigv = subparsers.add_parser("eigv", help="Compute EigV baseline scores.")
    eigv.add_argument("--config", type=Path, default=None)
    eigv.add_argument("--logs", type=Path, default=None)
    eigv.add_argument("--out", type=Path, default=None)
    eigv.add_argument("--mode", choices=["final", "whole"], default=None)
    eigv.add_argument("--model", default=None)
    eigv.add_argument("--score", choices=["sum", "spectral_entropy", "lambda_max"], default=None)
    eigv.add_argument("--device", default=None)
    eigv.set_defaults(func=run_eigv)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)
    args.func(args)


if __name__ == "__main__":
    main()
