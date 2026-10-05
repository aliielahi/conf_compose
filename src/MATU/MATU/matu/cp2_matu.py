"""Run MATU uncertainty scoring with PARAFAC2-style CP-2 decomposition.

The CLI accepts one or more role-specific embedding pickles created by
``matu/embed_logs.py``. For every task key, it builds a ragged matrix list from
the repeated runs and computes reconstruction losses over a rank range.
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
from tensorly.decomposition import parafac
from tqdm import tqdm


LOGGER = logging.getLogger(__name__)


def parafac2_als(
    x_list: list[np.ndarray],
    rank: int,
    *,
    max_iter: int = 50,
    tol: float = 1e-6,
    seed: int = 0,
    verbose: bool = False,
) -> tuple[list[np.ndarray], float, float, float]:
    """Approximate a ragged tensor represented as matrices with shared columns.

    Args:
        x_list: list of matrices with shapes (T_i, D).
        rank: target decomposition rank.
        max_iter: ALS iterations.
        tol: relative convergence tolerance.
        seed: random seed for initialization.
        verbose: print per-iteration objective.

    Returns:
        reconstructed matrices, fit, absolute reconstruction loss, relative loss.
    """
    if not x_list:
        raise ValueError("x_list is empty")

    matrices = [np.asarray(x, dtype=np.float64) for x in x_list if np.asarray(x).size > 0]
    if not matrices:
        raise ValueError("all matrices are empty")

    feature_dim = matrices[0].shape[1]
    for x in matrices:
        if x.ndim != 2:
            raise ValueError(f"expected a 2D matrix, got shape {x.shape}")
        if x.shape[1] != feature_dim:
            raise ValueError("all matrices must have the same embedding dimension")

    rng = np.random.default_rng(seed)
    n_slices = len(matrices)
    h = rng.standard_normal((rank, rank))
    v = rng.standard_normal((feature_dim, rank))
    s_list = [np.eye(rank) for _ in range(n_slices)]
    q_list: list[np.ndarray] = [np.empty((0, rank)) for _ in range(n_slices)]

    prev_obj = np.inf
    for iteration in range(max_iter):
        for i, x_i in enumerate(matrices):
            m_i = h @ s_list[i] @ (v.T @ x_i.T)
            u, _, vt = np.linalg.svd(m_i, full_matrices=False)
            q_list[i] = vt.T @ u.T

        y_tensor = np.zeros((rank, feature_dim, n_slices), dtype=np.float64)
        for i, x_i in enumerate(matrices):
            y_tensor[:, :, i] = q_list[i].T @ x_i

        w_init = rng.standard_normal((n_slices, rank))
        cp_init = (np.ones(rank), [h, v, w_init])
        cp_decomp = parafac(y_tensor, rank=rank, n_iter_max=1, init=cp_init, tol=tol, verbose=False)
        h, v, w = cp_decomp.factors

        for i in range(n_slices):
            s_list[i] = np.diag(w[i, :])

        obj_sq = 0.0
        for i, x_i in enumerate(matrices):
            x_hat_i = q_list[i] @ h @ s_list[i] @ v.T
            obj_sq += float(np.linalg.norm(x_i - x_hat_i) ** 2)
        obj = float(np.sqrt(obj_sq))

        if verbose:
            LOGGER.info("rank=%s iter=%s loss=%.6f", rank, iteration + 1, obj)
        if np.isfinite(prev_obj) and abs(prev_obj - obj) < tol * max(prev_obj, 1e-12):
            break
        prev_obj = obj

    reconstructions = []
    norm_x_sq = 0.0
    norm_res_sq = 0.0
    for i, x_i in enumerate(matrices):
        u_i = q_list[i] @ h
        x_hat_i = u_i @ s_list[i] @ v.T
        reconstructions.append(x_hat_i)
        norm_x_sq += float(np.linalg.norm(x_i) ** 2)
        norm_res_sq += float(np.linalg.norm(x_i - x_hat_i) ** 2)

    norm_x = float(np.sqrt(norm_x_sq))
    loss = float(np.sqrt(norm_res_sq))
    relative_loss = loss / max(norm_x, 1e-12)
    fit = 1.0 - relative_loss
    return reconstructions, fit, loss, relative_loss


def load_embedding_pickle(path: Path) -> dict[str, list[np.ndarray]]:
    with path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a dict[key -> list[np.ndarray]]")
    return data


def build_matrix_list(
    embedding_dicts: list[dict[str, list[np.ndarray]]],
    key: str,
    *,
    combine_mode: str,
    normalize: bool,
) -> list[np.ndarray]:
    role_runs = [d[key] for d in embedding_dicts if key in d]
    if not role_runs:
        return []

    matrices: list[np.ndarray] = []
    if combine_mode == "interleave":
        max_runs = max(len(runs) for runs in role_runs)
        for run_idx in range(max_runs):
            for runs in role_runs:
                if run_idx < len(runs):
                    matrices.append(np.asarray(runs[run_idx], dtype=np.float64))
    elif combine_mode == "concat_steps":
        max_runs = max(len(runs) for runs in role_runs)
        for run_idx in range(max_runs):
            pieces = [np.asarray(runs[run_idx], dtype=np.float64) for runs in role_runs if run_idx < len(runs)]
            pieces = [x for x in pieces if x.size > 0]
            if pieces:
                matrices.append(np.vstack(pieces))
    else:
        raise ValueError(f"unknown combine_mode: {combine_mode}")

    cleaned = []
    for matrix in matrices:
        if matrix.size == 0:
            continue
        if normalize:
            denom = np.linalg.norm(matrix)
            matrix = matrix / denom if denom > 0 else matrix
        cleaned.append(matrix)
    return cleaned


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    parser = argparse.ArgumentParser(description="Compute MATU scores from embedding matrices.")
    parser.add_argument("--embeddings", type=Path, nargs="+", required=True, help="Role embedding pickle(s).")
    parser.add_argument("--out", type=Path, required=True, help="Output pickle path.")
    parser.add_argument("--min_rank", type=int, default=1)
    parser.add_argument("--max_rank", type=int, default=50)
    parser.add_argument("--max_iter", type=int, default=25)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--combine_mode", choices=["interleave", "concat_steps"], default="interleave")
    parser.add_argument("--no_normalize", action="store_true", help="Disable per-matrix L2 normalization.")
    parser.add_argument("--legacy_fit_out", type=Path, default=None, help="Optional output matching old fit_dict format.")
    args = parser.parse_args()

    embedding_dicts = [load_embedding_pickle(path) for path in args.embeddings]
    keys = sorted(set.intersection(*(set(d.keys()) for d in embedding_dicts)))
    if not keys:
        raise ValueError("No overlapping task keys across embedding pickles.")

    ranks = list(range(args.min_rank, args.max_rank + 1))
    results = {}
    legacy_fit = {}

    for key in tqdm(keys, desc="MATU"):
        matrices = build_matrix_list(
            embedding_dicts,
            key,
            combine_mode=args.combine_mode,
            normalize=not args.no_normalize,
        )
        fit_values = []
        losses = []
        relative_losses = []

        for rank in ranks:
            try:
                _, fit, loss, rel_loss = parafac2_als(
                    matrices,
                    rank,
                    max_iter=args.max_iter,
                    tol=args.tol,
                    seed=args.seed,
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

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as f:
        pickle.dump(results, f, protocol=4)
    LOGGER.info("Saved MATU scores to %s", args.out)

    if args.legacy_fit_out is not None:
        args.legacy_fit_out.parent.mkdir(parents=True, exist_ok=True)
        with args.legacy_fit_out.open("wb") as f:
            pickle.dump(legacy_fit, f, protocol=4)
        LOGGER.info("Saved legacy fit_dict to %s", args.legacy_fit_out)


if __name__ == "__main__":
    main()
