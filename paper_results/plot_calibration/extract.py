"""Per-question raw confidences in the system's answer: the best single agent's and the log-odds pools'."""

import csv
import gzip
import json
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "paper_results/codes/voting_protocol"))

import data as voting  # noqa: E402
from conf_compose.composition.pooling import pool_methods  # noqa: E402
from conf_compose.data import Example, get_task  # noqa: E402

HERE = Path(__file__).resolve().parent
POOL_DIR = ROOT / "results/voting_atomic/learned_panels_confidence_tie_v2"
DEBATE_RUN = ROOT / "results/debate_composition/run_40bca54388122738"
STORE = ROOT / "results/inferences"
CORRECTIONS = ("shared_rho", "shared_scale", "logistic_pool")
FIELDS = ("protocol", "task", "models", "n_models", "method", "confidence", "correct")


def voting_rows():
    atomic = voting.atomic_module()
    sources = voting.SourceStore(STORE)
    for row in voting.load_csvs(POOL_DIR, ["cons"]):
        if row["fit_split"] != "holdout":
            raise ValueError(f"expected a holdout split: {row['task']}/{row['models']}")
        group, task = json.loads(row["model_ids"]), get_task(row["task"])
        args = voting.row_arguments(row, STORE)
        records = sources.get(atomic, row, group, "test")
        items = atomic.group_items(records, group)
        fitting, evaluation = atomic.split_items(row["task"], items, float(row["fit_fraction"]), int(row["fit_seed"]))
        accuracy = [mean(float(records[m][i.example_id]["correct"]) for i in fitting) for m in group]
        reference = accuracy.index(max(accuracy))
        fits = {name: voting.restore_fit(row, name) for name in CORRECTIONS}
        base = {"protocol": "voting", "task": row["task"], "models": row["models"], "n_models": len(group)}
        for item in evaluation:
            target, own, scores = atomic.confidence_scores(task, item, args, records)
            if target is None or not all(map(voting.valid_probability, [*own, *scores])):
                continue
            correct = float(task.is_correct(target, Example(item.example_id, item.question, item.gold)))
            yield {**base, "method": "single", "confidence": scores[reference], "correct": correct}
            yield {**base, "method": "logodds_sum", "confidence": pool_methods(scores)["logodds_sum"].score,
                   "correct": correct}
            for name, fit in fits.items():
                score = fit.predict(scores).score
                if score is not None:
                    yield {**base, "method": name, "confidence": score, "correct": correct}


def debate_rows():
    """Debate pools every member's round-0 and post-debate confidence in the final answer, as in the main table."""
    for path in sorted(DEBATE_RUN.glob("*/*/round_1/cons/predictions.jsonl")):
        audit = json.loads(path.with_name("audit.json").read_text())
        models = audit["models"].split("|")
        reference = models.index(audit["reference_model"])
        base = {"protocol": "debate", "task": audit["task"], "models": audit["models"], "n_models": len(models)}
        for line in path.open():
            record = json.loads(line)
            if not record["matched"]:
                continue
            correct = float(record["correct"])
            single = record["post_debate"]["shared"][reference]
            if single is not None:
                yield {**base, "method": "single", "confidence": single, "correct": correct}
            for name in ("logodds_sum", *CORRECTIONS):
                pooled = record["pooled"].get(f"both_rounds:{name}")
                if pooled and pooled["confidence"] is not None:
                    yield {**base, "method": name, "confidence": pooled["confidence"], "correct": correct}


def main():
    path = HERE / "confidences.csv.gz"
    with gzip.open(path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for source in (voting_rows, debate_rows):
            count = 0
            for row in source():
                writer.writerow(row)
                count += 1
            print(f"{source.__name__}: {count} rows", flush=True)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
