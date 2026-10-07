import hashlib
import json
from pathlib import Path

from data import SourceStore, atomic_module, digest, load_csvs, row_arguments
from tables import write_csv


def code_hash(root):
    paths = sorted((root / "src/conf_compose/composition").rglob("*.py"))
    paths += [root / "runs/experiment02-voting_composition/atomic.py", Path(__file__),
              root / "src/conf_compose/utils/metrics.py"]
    return hashlib.sha256(json.dumps([(str(path.relative_to(root)), digest(path)) for path in paths]).encode()).hexdigest()


def prepare_pools(source, destination, store, estimators, tasks, tie_break, tie_seed, ablations):
    if any((destination / f"{estimator}.csv").exists() for estimator in estimators):
        raise ValueError("refitting requires a fresh pool directory; saved fits are never overwritten")
    atomic = atomic_module()
    sources = SourceStore(store)
    cells = [row for row in load_csvs(source, estimators) if row["task"] in tasks]
    cells.sort(key=lambda row: (row["task"], row["models"], row["estimator"]))
    output = {estimator: [] for estimator in estimators}
    root = Path(__file__).resolve().parents[3]
    fingerprint = code_hash(root)
    for index, row in enumerate(cells, 1):
        group = json.loads(row["model_ids"])
        records = sources.get(atomic, row, group, "test")
        items = atomic.group_items(records, group)
        args = row_arguments(row, store)
        args.tie_break, args.tie_seed = tie_break, tie_seed
        args.fit_split = row["fit_split"]
        args.fit_fraction = float(row["fit_fraction"] or 0.3)
        args.fit_seed = int(row["fit_seed"] or 0)
        args.ablations = ablations
        args.logistic_l2 = float(row.get("logistic_l2") or 1)
        fitting, evaluation = atomic.split_items(row["task"], items, args.fit_fraction, args.fit_seed) if args.fit_split == "holdout" else (None, items)
        fitting_records = records
        if args.fit_split != "holdout":
            fitting_records = sources.get(atomic, row, group, "validation")
            fitting = atomic.group_items(fitting_records, group)
        for name, subset in (("fit", fitting), ("eval", evaluation)):
            if atomic.id_digest([item.example_id for item in subset]) != row[f"{name}_ids_hash"]:
                raise ValueError(f"refitting must preserve the original {name} IDs")
        result = atomic.run_task(row["task"], args, items, records,
                                 fitting if args.fit_split != "holdout" else None,
                                 fitting_records if args.fit_split != "holdout" else None)
        result.update(code_hash=fingerprint, input_hashes=row["input_hashes"],
                      source_pool=str(source / f"{row['estimator']}.csv"))
        output[row["estimator"]].append(result)
        print(f"Refit [{index}/{len(cells)}] {row['task']} {row['estimator']} {row['models']}", flush=True)
    destination.mkdir(parents=True, exist_ok=True)
    for estimator, rows in output.items():
        write_csv(destination / f"{estimator}.csv", rows)
    manifest = {"source": str(source), "source_hashes": {estimator: digest(source / f"{estimator}.csv") for estimator in estimators},
                "tie_break": tie_break, "tie_seed": tie_seed, "selection_signal": "consistency_t0.7",
                "code_hash": fingerprint, "fit_splits": "unchanged from source", "inference": False,
                "ablations": ablations, "n_cells": len(cells)}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
