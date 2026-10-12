"""One row per (estimator, protocol, dataset, group, system) with absolute p-Brier and accuracy-matched AUARC: the
paired observations every significance test uses. Every value is the one behind the paper's tables."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from conf_compose.utils import metrics
from conf_compose.utils.calibration import PlattCalibrator

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = ROOT / "paper_results/results"
PROTOCOLS = {"voting": "Before Interaction", "debate_r1": "Interaction Round 1", "debate_r2": "Interaction Round 2"}
DEBATE_RUNS = {"debate_r1": "run_40bca54388122738", "debate_r2": "run_12a013aed30bd201"}
# CAGE-CAL adapter run per protocol; round two is added here once its adapter is trained.
CAGECAL = {"voting": ROOT / "baselines/results/voting_adapter/a1fc8f63af8e71c5",
           "debate_r1": ROOT / "baselines/results/debate_adapter/71259088b2f0e984",
           "debate_r2": None}
# Weighting-study method -> system name, as the rows of the main table.
POOLING = {"weighted_average_1se": "Prob. Learned Weights", "logodds_sum": "Log-odds Sum",
           "shared_scale": "Fitted Scale", "weighted_sum_1se": "Log-odds Learned Weights"}
SEQUENCE = {"weighted_average": "Prob. Learned Weights", "weighted_sum": "Log-odds Learned Weights"}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def setting(protocol):
    """The weighting study's name for a protocol: debate rounds pool each agent's round-0 confidence."""
    return protocol if protocol == "voting" else f"{protocol}_both"


def pooling_rows():
    """Consistency pooling rules and the best agent, per group."""
    study = load("weighting_table", ROOT / "paper_results/weighting_study/table.py")
    scores = pd.read_csv(ROOT / "paper_results/weighting_study/scores.csv")
    references = study.references()
    rows = []
    for protocol in PROTOCOLS:
        found = scores[(scores.setting == setting(protocol)) & scores.method.isin(POOLING)]
        rows.append(pd.DataFrame({"protocol": protocol, "task": found.task, "models": found.models,
                                  "system": found.method.map(POOLING), "p_brier": found.p_brier, "auarc": found.auarc}))
        best = references[references.setting == setting(protocol)]
        rows.append(pd.DataFrame({"protocol": protocol, "task": best.task, "models": best.models, "system": "Best Agent",
                                  "p_brier": best.reference_p_brier, "auarc": best.reference_auarc}))
    return pd.concat(rows)


def audits(protocol):
    if protocol == "voting":
        return [a for a in json.loads((RESULTS / "voting_protocol/audit.json").read_text()) if a["estimator"] == "cons"]
    return json.loads((RESULTS / f"debate_protocol/{DEBATE_RUNS[protocol]}/cons/audit.json").read_text())


def cagecal_rows(protocol):
    """CAGE-CAL's ensemble score on each group's matched questions; p-Brier Platt-fitted on its own validation slice."""
    directory = CAGECAL[protocol]
    if directory is None:
        return pd.DataFrame()
    read = lambda name: {(r["task"], r["models"], r["id"]): r  # noqa: E731
                         for r in map(json.loads, (directory / name).open())}
    predictions, validation = read("predictions.jsonl"), read("validation_predictions.jsonl")
    rows = []
    for audit in audits(protocol):
        selected = [predictions[audit["task"], audit["models"], q] for q in audit["matched_ids"]]
        if any(p["target"] != audit["selected_targets"][p["id"]] for p in selected):
            raise ValueError(f"CAGE-CAL scored another answer: {protocol} {audit['task']} {audit['models']}")
        scores = np.asarray([p["raw"] for p in selected], dtype=float)
        labels = np.asarray([float(p["correct"]) for p in selected])
        own = [v for (task, models, _), v in validation.items() if (task, models) == (audit["task"], audit["models"])]
        calibrated = PlattCalibrator().fit([v["raw"] for v in own], [v["correct"] for v in own]).predict(scores)
        rows.append({"protocol": protocol, "task": audit["task"], "models": audit["models"], "system": "CAGE-CAL",
                     "p_brier": float(metrics.brier(calibrated, labels)),
                     "auarc": float(metrics.auarc(np.round(scores, 12), labels))})
    return pd.DataFrame(rows)


def judge_rows():
    """The LLM judge's verbalized confidence, as in the main table's LLM Elicited rows."""
    voting = pd.read_csv(RESULTS / "voting_protocol/atomic.csv")
    voting = voting[(voting.estimator == "cons") & (voting.method == "judge:g3-27i:reasoning_confidence")]
    debate = pd.read_csv(ROOT / "paper_results/judge_debate/judge_groups.csv")
    return pd.concat([
        pd.DataFrame({"protocol": "voting", "task": voting.task, "models": voting.models, "system": "LLM Elicited",
                      "p_brier": voting.p_brier, "auarc": voting.answer_matched_auarc}),
        pd.DataFrame({"protocol": "debate_r" + debate["round"].astype(str), "task": debate.task, "models": debate.models,
                      "system": "LLM Elicited", "p_brier": debate.p_brier, "auarc": debate.auarc})])


def sequence_rows():
    """Sequence probability (after reasoning): both learned-weight rules and the best agent, per group."""
    frame = pd.read_csv(ROOT / "paper_results/per_system_results/systems.csv")
    frame = frame[frame.estimator == "seq_reasoned"]
    rules = pd.DataFrame({"protocol": frame.setting, "task": frame.task, "models": frame.models,
                          "system": frame.method.map(SEQUENCE), "p_brier": frame.p_brier, "auarc": frame.auarc})
    best = frame.drop_duplicates(["setting", "task", "models"])
    best = pd.DataFrame({"protocol": best.setting, "task": best.task, "models": best.models, "system": "Best Agent",
                         "p_brier": best.reference_p_brier, "auarc": best.reference_auarc})
    return pd.concat([rules, best])


def check_cagecal(frame):
    """Recomputed CAGE-CAL must equal the pipeline's own table rows wherever the pipeline has them."""
    tables = {"voting": (RESULTS / "voting_protocol/atomic.csv", "cagecal_iid"),
              **{p: (RESULTS / f"debate_protocol/{run}/cons/atomic.csv", "cagecal_debate") for p, run in DEBATE_RUNS.items()}}
    for protocol, (path, method) in tables.items():
        table = pd.read_csv(path)
        table = table[(table.method == method) & (table.get("estimator", "cons") == "cons")]
        mine = frame[(frame.protocol == protocol) & (frame.system == "CAGE-CAL")]
        if table.empty or mine.empty:
            continue
        joined = mine.merge(table, on=["task", "models"], suffixes=("", "_table"), validate="one_to_one")
        gap = max((joined.p_brier - joined.p_brier_table).abs().max(),
                  (joined.auarc - joined.answer_matched_auarc).abs().max())
        if len(joined) != 75 or gap > 1e-9:
            raise ValueError(f"CAGE-CAL {protocol} does not reproduce the table (groups {len(joined)}, gap {gap})")


def build():
    consistency = pd.concat([pooling_rows(), judge_rows(), *[cagecal_rows(p) for p in PROTOCOLS]], ignore_index=True)
    check_cagecal(consistency)
    frame = pd.concat([consistency.assign(estimator="consistency"), sequence_rows().assign(estimator="sequence")],
                      ignore_index=True)
    groups = {frozenset(p): number for number, p in
              enumerate(json.loads((ROOT / "src/conf_compose/constants.json").read_text())["composition_panels"], 1)}
    frame["group"] = frame.models.map(lambda m: groups[frozenset(m.split("|"))])
    counts = frame.groupby(["estimator", "protocol", "system"]).size()
    if (counts != 75).any():
        raise ValueError(f"every system needs 75 blocks per protocol:\n{counts[counts != 75]}")
    return frame[["estimator", "protocol", "task", "group", "models", "system", "p_brier", "auarc"]]


if __name__ == "__main__":
    built = build()
    built.to_csv(HERE / "blocks.csv", index=False)
    print(built.groupby(["estimator", "protocol", "system"]).size().unstack("protocol").to_string())
