"""Per-group fitting and evaluation confidence matrices for voting and both debate rounds, saved as cells.pkl."""

import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "paper_results/codes/voting_protocol"))

import data as voting  # noqa: E402
from conf_compose.composition.pooling import pool_methods  # noqa: E402
from conf_compose.data import Example, get_task  # noqa: E402

HERE = Path(__file__).resolve().parent
POOL_DIR = ROOT / "results/voting_atomic/learned_panels_confidence_tie_v2"
STORE = ROOT / "results/inferences"
VOTING_AUDIT = ROOT / "paper_results/results/voting_protocol/audit.json"
DEBATE_RUNS = {1: ROOT / "results/debate_composition/run_40bca54388122738",
               2: ROOT / "results/debate_composition/run_12a013aed30bd201"}
# The pipeline's own outputs, read back rather than refitted, so they match the paper table exactly.
PIPELINE = ("mean", "logodds_sum", "blp", "shared_scale", "logistic_pool", "kahn", "kahn_diagonal")


def cell(setting, task, models, fit, evaluation):
    """`fit` and `evaluation` are lists of (streams, label, {method: (confidence, ranking)})."""
    def arrays(rows):
        return (np.asarray([r[0] for r in rows], dtype=float), np.asarray([r[1] for r in rows], dtype=float),
                {m: np.asarray([r[2][m] for r in rows], dtype=float) for m in PIPELINE})
    fit_p, fit_y, fit_out = arrays(fit)
    eval_p, eval_y, eval_out = arrays(evaluation)
    return {"setting": setting, "task": task, "models": models, "n_models": len(models.split("|")),
            "fit": fit_p, "fit_labels": fit_y, "eval": eval_p, "eval_labels": eval_y,
            "pipeline": {m: (fit_out[m][:, 0], eval_out[m][:, 0], eval_out[m][:, 1]) for m in PIPELINE}}


def voting_cells():
    """Evaluation questions are the pipeline's matched ones, which also drop questions a judge did not score."""
    matched = {(cell["task"], cell["models"]): set(cell["matched_ids"])
               for cell in json.loads(VOTING_AUDIT.read_text()) if cell["estimator"] == "cons"}
    atomic = voting.atomic_module()
    sources = voting.SourceStore(STORE)
    for row in voting.load_csvs(POOL_DIR, ["cons"]):
        group, task = json.loads(row["model_ids"]), get_task(row["task"])
        args = voting.row_arguments(row, STORE)
        records = sources.get(atomic, row, group, "test")
        items = atomic.group_items(records, group)
        fitting, evaluation = atomic.split_items(row["task"], items, float(row["fit_fraction"]), int(row["fit_seed"]))
        fits = {m: voting.restore_fit(row, m) for m in PIPELINE if m not in ("mean", "logodds_sum")}
        matrices = {"fit": [], "eval": []}
        for name, subset in (("fit", fitting), ("eval", evaluation)):
            for item in subset:
                target, own, scores = atomic.confidence_scores(task, item, args, records)
                if target is None or not all(map(voting.valid_probability, scores)):
                    continue
                if name == "eval" and item.example_id not in matched[row["task"], row["models"]]:
                    continue
                predictions = {**pool_methods(scores), **{m: fit.predict(scores) for m, fit in fits.items()}}
                outputs = {m: (predictions[m].score, voting.rank_value(predictions[m])) for m in PIPELINE}
                label = float(task.is_correct(target, Example(item.example_id, item.question, item.gold)))
                matrices[name].append((scores, label, outputs))
        yield cell("voting", row["task"], row["models"], matrices["fit"], matrices["eval"])


def debate_cells(round_index, run):
    """Post-debate streams, and post-debate plus round-0 streams, both in the final answer, as in the pipeline."""
    for path in sorted(run.glob(f"*/*/round_{round_index}/cons/predictions.jsonl")):
        audit = json.loads(path.with_name("audit.json").read_text())
        records = {record["id"]: record for record in map(json.loads, path.open())}

        def streams(record, both):
            values = record["post_debate"]["shared"] + (record["initial_on_final"] if both else [])
            return values if all(value is not None for value in values) else None

        for both, prefix in ((False, ""), (True, "both_rounds:")):
            rows = {}
            for name, ids in (("fit", audit["fit_scored_ids"]), ("eval", audit["matched_ids"])):
                kept = [records[q] for q in ids if streams(records[q], both) is not None]
                rows[name] = [(streams(r, both), float(r["correct"]),
                               {m: (r["pooled"][prefix + m]["confidence"], r["pooled"][prefix + m]["ranking"])
                                for m in PIPELINE}) for r in kept]
            setting = f"debate_r{round_index}" + ("_both" if both else "")
            yield cell(setting, audit["task"], audit["models"], rows["fit"], rows["eval"])


def main():
    cells = list(voting_cells())
    for round_index, run in DEBATE_RUNS.items():
        cells += list(debate_cells(round_index, run))
    with (HERE / "cells.pkl").open("wb") as handle:
        pickle.dump(cells, handle)
    counts = {}
    for item in cells:
        counts[item["setting"]] = counts.get(item["setting"], 0) + 1
    print(counts)


if __name__ == "__main__":
    main()
