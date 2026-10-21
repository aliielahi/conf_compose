"""Is composition better than the single agent with the best confidence, not only the most accurate one? Every agent's
confidence in the selected answer is scored on the same questions as log-odds Learned Weights."""

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


def agent_scores(cell):
    """Per agent (post-interaction stream after debate): validation and test AUARC and p-Brier, plus validation accuracy proxy."""
    warnings.filterwarnings("ignore")
    k = cell["n_models"]
    fit, ev, fy, ey = cell["fit"][:, :k], cell["eval"][:, :k], cell["fit_labels"], cell["eval_labels"]
    rows = []
    for i in range(k):
        f, e = np.round(fit[:, i], 12), np.round(ev[:, i], 12)
        platt = PlattCalibrator()
        platt.fit(f, fy)
        # Validation scores use out-of-sample-free in-sample Platt; only used for choosing an agent.
        rows.append({"agent": i, "fit_auarc": float(metrics.auarc(f, fy)), "fit_brier": float(metrics.brier(platt.predict(f), fy)),
                     "auarc": float(metrics.auarc(e, ey)), "p_brier": float(metrics.brier(platt.predict(e), ey))})
    frame = pd.DataFrame(rows)
    out = {"estimator": cell["estimator"], "setting": cell["setting"], "task": cell["task"], "models": cell["models"]}
    out["valbest_auarc"] = frame.loc[frame.fit_auarc.idxmax(), "auarc"]
    out["valbest_brier"] = frame.loc[frame.fit_brier.idxmin(), "p_brier"]
    out["oracle_auarc"] = frame.auarc.max()
    out["oracle_brier"] = frame.p_brier.min()
    return out


def main():
    cells = pickle.loads((ROOT / "paper_results/logodds_study/cells.pkl").read_bytes())
    with Pool() as pool:
        refs = pd.DataFrame(pool.map(agent_scores, cells))
    raw = pd.read_csv(ROOT / "paper_results/raw_calibration/raw.csv")
    systems = pd.read_csv(ROOT / "paper_results/per_system_results/systems.csv")
    acc = systems[systems.method == "weighted_sum"][["estimator", "setting", "task", "models", "reference_auarc", "reference_p_brier"]]
    frame = raw.merge(refs, on=["estimator", "setting", "task", "models"]).merge(acc, on=["estimator", "setting", "task", "models"])
    frame.to_csv(HERE / "references.csv", index=False)
    tasks = ["csqa", "boolq", "gsm8k", "truthfulqa", "gpqa"]
    for est in ("cons", "seq_reasoned"):
        x = frame[frame.estimator == est]
        print(f"\n===== {est}: log-odds Learned Weights minus reference (x100); AUARC positive and p-Brier negative favor composition")
        for name, (a_ref, b_ref) in {"accuracy-selected (paper)": ("reference_auarc", "reference_p_brier"),
                                     "selected by validation confidence": ("valbest_auarc", "valbest_brier"),
                                     "oracle best on test": ("oracle_auarc", "oracle_brier")}.items():
            da, db = 100 * (x.auarc - x[a_ref]), 100 * (x.p_brier - x[b_ref])
            cell = pd.DataFrame({"a": da, "b": db, "setting": x.setting, "task": x.task}).groupby(["setting", "task"]).mean()
            print(f"{name:36} AUARC {da.mean():+.2f} (wins {(da > 0).sum()}/{len(da)}, cells {(cell.a > 0).sum()}/15) | "
                  f"p-Brier {db.mean():+.2f} (wins {(db < 0).sum()}/{len(db)}, cells {(cell.b < 0).sum()}/15)")


if __name__ == "__main__":
    main()
