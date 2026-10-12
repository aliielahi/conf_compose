"""Absolute per-group results (matched accuracy, AUARC, p-Brier) for the best agent and both learned-weight rules."""

import importlib.util
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
# The seq-vs-cons replay of the voting and debate pipelines, loaded by path since both files are named extract.py.
_spec = importlib.util.spec_from_file_location("seq_vs_cons_extract", HERE.parent / "seq_vs_cons" / "extract.py")
replay = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(replay)

LEARNED = ("weighted_sum", "weighted_average")
# Confidence estimator -> (pipeline estimator, scoring context, sequence score).
ESTIMATORS = {"cons": ("cons", "direct", "norm_sum"), "seq_reasoned": ("seq", "reasoned", "norm_sum")}
COLUMNS = ["estimator", "setting", "task", "models", "n_models", "n_eval", "method", "accuracy", "auarc",
           "reference_auarc", "p_brier", "reference_p_brier"]


def job(arguments):
    name, setting = arguments
    estimator, context, seq_score = ESTIMATORS[name]
    if setting == "voting":
        rows = replay.voting_rows(estimator, context, seq_score, LEARNED)
    else:
        round_index = int(setting[-1])
        run = (replay.CONSISTENCY_RUNS[round_index] if estimator == "cons"
               else replay.sequence_runs()[context, seq_score])
        rows = replay.debate_rows(run, round_index, estimator, LEARNED)
    frame = pd.DataFrame(rows).assign(estimator=name, setting=setting)
    print(f"{name} {setting}: {frame.models.nunique()} groups", flush=True)
    return frame


def main():
    jobs = [(name, setting) for name in ESTIMATORS for setting in ("voting", "debate_r1", "debate_r2")]
    with Pool(len(jobs)) as pool:
        frame = pd.concat(pool.map(job, jobs), ignore_index=True)
    frame = frame[frame.method.isin(LEARNED)][COLUMNS]
    frame.to_csv(HERE / "systems.csv", index=False)
    print(f"wrote {HERE / 'systems.csv'} ({len(frame)} rows)")


if __name__ == "__main__":
    main()
