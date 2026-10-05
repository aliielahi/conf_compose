"""Convert CP-2 fit curves into scalar MATU uncertainty scores.

The original experiment scripts save ``fit_dict``:

    dict[task_id, list[fit_rank_1, ..., fit_rank_R]]

where ``fit = 1 - relative_reconstruction_loss``. Therefore the MATU
uncertainty used for evaluation can be recovered as:

    U = sum_R (1 - fit_R)

This script makes that conversion explicit and saves:

    dict[task_id, float]
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np


LOGGER = logging.getLogger(__name__)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def save_pickle(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=4)


def uncertainty_from_legacy_fit(fit_values: Any) -> float:
    fit = np.asarray(fit_values, dtype=float)
    return float(np.nansum(1.0 - fit))


def uncertainty_from_matu_result(value: Any) -> float:
    if isinstance(value, dict):
        if "uncertainty" in value:
            return float(value["uncertainty"])
        if "relative_loss" in value:
            return float(np.nansum(np.asarray(value["relative_loss"], dtype=float)))
        if "fit" in value:
            return uncertainty_from_legacy_fit(value["fit"])
    return uncertainty_from_legacy_fit(value)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    parser = argparse.ArgumentParser(description="Convert fit_dict / MATU result pickle to scalar uncertainty.")
    parser.add_argument("--fit_dict", type=Path, required=True, help="Legacy fit_dict or cp2_matu output pickle.")
    parser.add_argument("--out", type=Path, required=True, help="Output uncertainty dict pickle.")
    args = parser.parse_args()

    fit_dict = load_pickle(args.fit_dict)
    if not isinstance(fit_dict, dict):
        raise TypeError("--fit_dict must contain a dict")

    uncertainty = {key: uncertainty_from_matu_result(value) for key, value in fit_dict.items()}
    save_pickle(uncertainty, args.out)
    LOGGER.info("Saved uncertainty dict to %s", args.out)


if __name__ == "__main__":
    main()
