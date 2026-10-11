"""Per-group Δp-Brier and ΔAUARC of the plain and the learned-weight log-odds sum, per confidence estimator."""

import json
import math
import sys
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "paper_results/codes/voting_protocol"))
sys.path.insert(0, str(ROOT / "paper_results/weighting_study"))

import data as voting  # noqa: E402
from methods import weighted_sum  # noqa: E402
from conf_compose.composition.pooling import pool_methods  # noqa: E402
from conf_compose.data import Example, get_task  # noqa: E402
from conf_compose.utils.calibration import PlattCalibrator  # noqa: E402
from conf_compose.utils.metrics import auarc, brier  # noqa: E402

HERE = Path(__file__).resolve().parent
POOL_DIR = ROOT / "results/voting_atomic/learned_panels_confidence_tie_v2"
VOTING_AUDIT = ROOT / "paper_results/results/voting_protocol/audit.json"
STORE = ROOT / "results/inferences"
DEBATE_DIR = ROOT / "results/debate_composition"
# Consistency runs already reported in the paper, one per debate round.
CONSISTENCY_RUNS = {1: "run_40bca54388122738", 2: "run_12a013aed30bd201"}
# Sequence-probability variants: scoring context and how candidate log-probs are combined before normalizing
# over every candidate answer (all labels on closed-label tasks).
VARIANTS = {"seq_label": ("direct", "norm_sum"), "seq_token": ("direct", "norm_mean"),
            "seq_reasoned": ("reasoned", "norm_sum")}


# The proposed learned weights: shrunk toward one shared scale, strength chosen by cross-validation (one-SE rule).
WEIGHTED = weighted_sum(one_se=True)


def methods(fit, fit_labels, evaluation, labels):
    """Both pooling rules as (Platt Brier, AUARC), from member confidence matrices."""
    fit, evaluation, fit_labels = (np.asarray(v, dtype=float) for v in (fit, evaluation, fit_labels))
    fit_conf, conf, rank = WEIGHTED(fit, fit_labels, evaluation)
    plain = [pool_methods(list(row))["logodds_sum"] for row in evaluation]
    return {"logodds_sum": (platt_brier([pool_methods(list(row))["logodds_sum"].score for row in fit], fit_labels,
                                        [p.score for p in plain], labels),
                            ranking_auarc([voting.rank_value(p) for p in plain], labels)),
            "weighted_sum": (platt_brier(fit_conf, fit_labels, conf, labels), ranking_auarc(rank, labels))}


def platt_brier(fit_scores, fit_labels, scores, labels):
    return float(brier(PlattCalibrator().fit(fit_scores, fit_labels).predict(scores), labels))


def ranking_auarc(ranks, labels):
    return float(auarc(np.round(np.asarray(ranks, dtype=float), 12), labels))


def voting_rows(estimator, context="direct", seq_score="norm_sum"):
    """Replays the voting pipeline's reference and log-odds sum on its own matched questions."""
    matched = {(cell["task"], cell["models"]): set(cell["matched_ids"])
               for cell in json.loads(VOTING_AUDIT.read_text()) if cell["estimator"] == estimator}
    atomic = voting.atomic_module()
    sources = voting.SourceStore(STORE)
    for row in voting.load_csvs(POOL_DIR, [estimator]):
        group, task = json.loads(row["model_ids"]), get_task(row["task"])
        args = voting.row_arguments(row, STORE)
        args.context, args.seq_score = context, seq_score
        records = sources.get(atomic, row, group, "test")
        items = atomic.group_items(records, group)
        fitting, evaluation = atomic.split_items(row["task"], items, float(row["fit_fraction"]), int(row["fit_seed"]))
        accuracy = [mean(float(records[m][i.example_id]["correct"]) for i in fitting) for m in group]
        reference = accuracy.index(max(accuracy))
        model = group[reference]
        fit_scores, fit_sum_labels, fit_own, fit_own_labels = [], [], [], []
        for item in fitting:
            target, own, scores = atomic.confidence_scores(task, item, args, records)
            if target is not None and all(map(voting.valid_probability, scores)):
                fit_scores.append(scores)
                fit_sum_labels.append(float(task.is_correct(target, Example(item.example_id, item.question, item.gold))))
            if all(map(voting.valid_probability, own)):
                fit_own.append(own[reference])
                fit_own_labels.append(float(records[model][item.example_id]["correct"]))
        matrix, labels, own_values, own_labels, shared = [], [], [], [], []
        for item in evaluation:
            if item.example_id not in matched[row["task"], row["models"]]:
                continue
            target, own, scores = atomic.confidence_scores(task, item, args, records)
            if target is None or not all(map(voting.valid_probability, [*scores, *own])):
                continue
            matrix.append(scores)
            labels.append(float(task.is_correct(target, Example(item.example_id, item.question, item.gold))))
            own_values.append(own[reference])
            own_labels.append(float(records[model][item.example_id]["correct"]))
            shared.append(scores[reference])
        reference_brier = platt_brier(fit_own, fit_own_labels, own_values, own_labels)
        reference_auarc = ranking_auarc(shared, labels)
        for method, (p_brier, auarc_value) in methods(fit_scores, fit_sum_labels, matrix, labels).items():
            yield {"task": row["task"], "models": row["models"], "n_models": len(group), "n_eval": len(labels),
                   "method": method, "delta_p_brier": p_brier - reference_brier,
                   "delta_auarc": auarc_value - reference_auarc}


def debate_rows(run, round_index, estimator):
    """Log-odds sum over round 0 and the given round against the best agent's post-debate confidence."""
    for path in sorted((DEBATE_DIR / run).glob(f"*/*/round_{round_index}/{estimator}/predictions.jsonl")):
        audit = json.loads(path.with_name("audit.json").read_text())
        records = {record["id"]: record for record in map(json.loads, path.open())}
        index = audit["models"].split("|").index(audit["reference_model"])

        def streams(question):
            values = records[question]["post_debate"]["shared"] + records[question]["initial_on_final"]
            return values if all(value is not None for value in values) else None

        def reference(question):
            return records[question]["post_debate"]["shared"][index]

        fit = [q for q in audit["fit_scored_ids"] if streams(q) is not None]
        fit_reference = [q for q in audit["fit_scored_ids"] if reference(q) is not None]
        both = [q for q in audit["matched_ids"] if streams(q) is not None]
        label = {q: float(records[q]["correct"]) for q in records}
        reference_brier = platt_brier([reference(q) for q in fit_reference], [label[q] for q in fit_reference],
                                      [reference(q) for q in audit["matched_ids"]], [label[q] for q in audit["matched_ids"]])
        reference_auarc = ranking_auarc([reference(q) for q in audit["matched_ids"]], [label[q] for q in audit["matched_ids"]])
        results = methods([streams(q) for q in fit], [label[q] for q in fit], [streams(q) for q in both],
                          [label[q] for q in both])
        for method, (p_brier, auarc_value) in results.items():
            yield {"task": audit["task"], "models": audit["models"], "n_models": audit["n_models"], "n_eval": len(both),
                   "method": method, "delta_p_brier": p_brier - reference_brier,
                   "delta_auarc": auarc_value - reference_auarc}


def sequence_runs():
    """Completed sequence-only composition runs, keyed by (context, seq_score)."""
    runs = {}
    for manifest in DEBATE_DIR.glob("run_*/manifest.json"):
        content = json.loads(manifest.read_text())
        identity = content["identity"]
        if identity["estimators"] == ["seq"] and len(identity.get("cells", [])) >= 150:
            runs[identity["config"]["context"], identity["config"]["seq_score"]] = manifest.parent.name
    return runs


def main():
    frames = [pd.DataFrame(voting_rows("cons")).assign(estimator="cons", setting="voting")]
    for round_index, run in CONSISTENCY_RUNS.items():
        frames.append(pd.DataFrame(debate_rows(run, round_index, "cons")).assign(estimator="cons", setting=f"debate_r{round_index}"))
    runs = sequence_runs()
    for name, (context, seq_score) in VARIANTS.items():
        frames.append(pd.DataFrame(voting_rows("seq", context, seq_score)).assign(estimator=name, setting="voting"))
        run = runs.get((context, seq_score))
        if run is None:
            print(f"{name}: no completed debate run yet, voting only", flush=True)
            continue
        for round_index in (1, 2):
            frames.append(pd.DataFrame(debate_rows(run, round_index, "seq")).assign(estimator=name, setting=f"debate_r{round_index}"))
    frame = pd.concat(frames, ignore_index=True)
    frame.to_csv(HERE / "deltas.csv", index=False)
    print(frame.groupby(["estimator", "setting", "method"]).size().unstack("method").to_string())


if __name__ == "__main__":
    main()
