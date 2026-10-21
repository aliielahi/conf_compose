"""Raw (no Platt step) ECE and Brier of log-odds Learned Weights per group, for both confidence estimators, with the
table's p-Brier and AUARC alongside as a check against per_system_results/systems.csv."""

import importlib.util
import pickle
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from conf_compose.utils import metrics
from conf_compose.utils.calibration import PlattCalibrator

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
_spec = importlib.util.spec_from_file_location("weighting_methods", ROOT / "paper_results/weighting_study/methods.py")
methods = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(methods)
RULE = methods.weighted_sum(one_se=True)


def score(cell):
    warnings.filterwarnings("ignore")
    fit_conf, conf, rank = (np.round(np.asarray(v, dtype=float), 12) for v in
                            RULE(cell["fit"], cell["fit_labels"], cell["eval"]))
    y = cell["eval_labels"]
    platt = PlattCalibrator().fit(fit_conf, cell["fit_labels"]).predict(conf)
    return {"estimator": cell["estimator"], "setting": cell["setting"], "task": cell["task"], "models": cell["models"],
            "n_models": cell["n_models"], "ece": float(metrics.ece(conf, y)), "brier": float(metrics.brier(conf, y)),
            "p_brier": float(metrics.brier(platt, y)), "auarc": float(metrics.auarc(rank, y))}


def main():
    cells = pickle.loads((ROOT / "paper_results/logodds_study/cells.pkl").read_bytes())
    with Pool() as pool:
        frame = pd.DataFrame(pool.map(score, cells))
    systems = pd.read_csv(ROOT / "paper_results/per_system_results/systems.csv")
    systems = systems[systems.method == "weighted_sum"]
    joined = frame.merge(systems, on=["estimator", "setting", "task", "models"], suffixes=("", "_table"),
                         validate="one_to_one")
    gap = max((joined.p_brier - joined.p_brier_table).abs().max(), (joined.auarc - joined.auarc_table).abs().max())
    if len(joined) != 450 or gap > 1e-6:
        raise ValueError(f"does not reproduce systems.csv: {len(joined)} groups, gap {gap}")
    frame.to_csv(HERE / "raw.csv", index=False)
    print(f"wrote {HERE / 'raw.csv'}; matches systems.csv (max gap {gap:.1e})")


if __name__ == "__main__":
    main()
