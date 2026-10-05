"""Step 2: run MATU/CP-2 on quick-start embedding matrices."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from env_utils import load_quickstart_env


QUICK_START = Path(__file__).resolve().parents[1]
REPO_ROOT = QUICK_START.parent
EMB_DIR = QUICK_START / "generated" / "embeddings"
REFERENCE_EMB_DIR = QUICK_START / "generated" / "reference_embeddings" / "math"
REFERENCE_ZIP = QUICK_START / "data" / "embeddings_Math_qwen2.5_qwen3.zip"
OUT_DIR = QUICK_START / "generated" / "results"
REFERENCE_EMBEDDINGS = (
    (
        "user_embedding_matrices_Math_qwen2.5_qwen3.pkl",
        REFERENCE_EMB_DIR / "user_embedding_matrices_Math_qwen2.5_qwen3.pkl",
    ),
    (
        "assistant_embedding_matrices_Math_qwen2.5_qwen3.pkl",
        REFERENCE_EMB_DIR / "assistant_embedding_matrices_Math_qwen2.5_qwen3.pkl",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run MATU/CP-2 on quick-start embeddings. By default this uses "
            "quick_start/generated/embeddings/*.pkl, which can be produced by "
            "01_embed_reference_logs.py."
        )
    )
    parser.add_argument(
        "--embedding-source",
        "--embedding_source",
        dest="embedding_source",
        choices=["generated", "provided"],
        default="generated",
        help=(
            "Embedding source to use when --embeddings is not set. "
            "'generated' reads quick_start/generated/embeddings/*.pkl; "
            "'provided' stages the packaged MATH embeddings from quick_start/data/."
        ),
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        nargs="+",
        default=None,
        help="Explicit role embedding pickle(s). Overrides --embedding-source.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT_DIR / "matu_scores.pkl",
        help="Output pickle path for structured MATU scores.",
    )
    parser.add_argument(
        "--legacy-fit-out",
        "--legacy_fit_out",
        dest="legacy_fit_out",
        type=Path,
        default=OUT_DIR / "fit_dict_generated.pkl",
        help="Output pickle path for the legacy fit_dict format.",
    )
    parser.add_argument("--min-rank", "--min_rank", dest="min_rank", type=int, default=1)
    parser.add_argument(
        "--max-rank",
        "--max_rank",
        dest="max_rank",
        type=int,
        default=None,
        help="Maximum CP-2 rank. Defaults to MATU_MAX_RANK or 50.",
    )
    parser.add_argument("--max-iter", "--max_iter", dest="max_iter", type=int, default=25)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--combine-mode",
        "--combine_mode",
        dest="combine_mode",
        choices=["interleave", "concat_steps"],
        default="interleave",
    )
    parser.add_argument(
        "--no-normalize",
        "--no_normalize",
        dest="no_normalize",
        action="store_true",
        help="Disable per-matrix L2 normalization.",
    )
    return parser.parse_args()


def stage_provided_embeddings() -> list[Path]:
    """Extract packaged MATH embeddings into generated/reference_embeddings."""
    REFERENCE_EMB_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(REFERENCE_ZIP) as archive:
        for member, out_path in REFERENCE_EMBEDDINGS:
            if out_path.exists():
                continue
            with archive.open(member) as src, out_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    print(f"Staged packaged MATH embeddings in {REFERENCE_EMB_DIR}.")
    return [out_path for _, out_path in REFERENCE_EMBEDDINGS]


def resolve_cli_path(path: Path) -> Path:
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def generated_embeddings() -> list[Path]:
    return [
        EMB_DIR / "user_embedding_matrices.pkl",
        EMB_DIR / "assistant_embedding_matrices.pkl",
    ]


def resolve_embeddings(args: argparse.Namespace) -> list[Path]:
    if args.embeddings is not None:
        return [resolve_cli_path(path) for path in args.embeddings]
    if args.embedding_source == "provided":
        return stage_provided_embeddings()
    return generated_embeddings()


def main() -> None:
    args = parse_args()
    load_quickstart_env()
    embeddings = resolve_embeddings(args)
    missing = [path for path in embeddings if not path.exists()]
    if missing:
        missing_text = "\n".join(f"  - {path}" for path in missing)
        raise SystemExit(
            "Missing embedding pickle(s):\n"
            f"{missing_text}\n"
            "Run quick_start/code/01_embed_reference_logs.py, use "
            "--embedding-source provided, or pass explicit --embeddings paths."
        )

    out_path = resolve_cli_path(args.out)
    legacy_fit_out = (
        resolve_cli_path(args.legacy_fit_out)
        if args.legacy_fit_out is not None
        else None
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if legacy_fit_out is not None:
        legacy_fit_out.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    max_rank = (
        args.max_rank
        if args.max_rank is not None
        else os.environ.get("MATU_MAX_RANK", "50")
    )

    cmd = [
        sys.executable,
        "-m",
        "matu.cp2_matu",
        "--embeddings",
        *[str(path) for path in embeddings],
        "--out",
        str(out_path),
        "--min_rank",
        str(args.min_rank),
        "--max_rank",
        str(max_rank),
        "--max_iter",
        str(args.max_iter),
        "--tol",
        str(args.tol),
        "--seed",
        str(args.seed),
        "--combine_mode",
        args.combine_mode,
    ]
    if args.no_normalize:
        cmd.append("--no_normalize")
    if legacy_fit_out is not None:
        cmd.extend(["--legacy_fit_out", str(legacy_fit_out)])
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), env=env)


if __name__ == "__main__":
    main()
