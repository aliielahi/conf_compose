import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from data import ABLATIONS, METHODS, SourceStore, digest, index_judges, load_csvs, process_cell
from significance import dataset_sign_tests
from tables import write_csv, write_significance_tables, write_tables
from prepare import prepare_pools


def parse_args():
    parser = argparse.ArgumentParser(description="Voting confidence tables with frozen saved fits or an explicit offline refit")
    parser.add_argument("--pool-dir", type=Path, default=ROOT / "results/voting_atomic/learned_panels_confidence_tie_v2")
    parser.add_argument("--source-pool-dir", type=Path, default=ROOT / "results/voting_atomic/learned_panels_v1")
    parser.add_argument("--refit", action="store_true", help="refit all pools on unchanged fitting IDs into a fresh pool directory")
    parser.add_argument("--tie-break", choices=("first", "confidence"), default="confidence")
    parser.add_argument("--tie-seed", type=int, default=0)
    parser.add_argument("--judge-policy", choices=("strict", "approximate"), default="approximate")
    parser.add_argument("--judge-dir", type=Path, default=ROOT / "results/experiment04-judge_baseline")
    parser.add_argument("--store", type=Path, default=ROOT / "results/inferences")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "paper_results/results/voting_protocol")
    parser.add_argument("--estimators", nargs="+", choices=("cons", "seq"), default=["cons", "seq"])
    parser.add_argument("--tasks", nargs="+", default=["csqa", "boolq", "gsm8k", "truthfulqa", "gpqa"])
    parser.add_argument("--judges", nargs="+", default=["g3-27i", "l32-3bi"])
    parser.add_argument("--reference", choices=("fit_metric", "fit_accuracy", "eval_accuracy", "metric_best"), default="fit_accuracy")
    parser.add_argument("--ablations", action="store_true")
    parser.add_argument("--significance", action="store_true", help="exploratory dataset-block sign tests with joint Holm correction")
    return parser.parse_args()


def main():
    args = parse_args()
    for key in ("pool_dir", "source_pool_dir", "judge_dir", "store", "out_dir"):
        setattr(args, key, getattr(args, key).resolve())
    args.estimators = list(dict.fromkeys(args.estimators))
    args.tasks = list(dict.fromkeys(args.tasks))
    args.judges = list(dict.fromkeys(judge.split("/")[-1] for judge in args.judges))
    if not args.significance and (args.out_dir / "significance.csv").exists():
        raise SystemExit("an earlier significance.csv exists; use a fresh --out-dir or include --significance")
    if args.refit:
        prepare_pools(args.source_pool_dir, args.pool_dir, args.store, args.estimators, args.tasks,
                      args.tie_break, args.tie_seed, args.ablations)
    cells = [cell for cell in load_csvs(args.pool_dir, args.estimators) if cell["task"] in args.tasks]
    if not cells:
        raise SystemExit("no matching pooling results")
    if any(cell.get("tie_break", "first") != args.tie_break or int(cell.get("tie_seed", 0)) != args.tie_seed for cell in cells):
        raise SystemExit("saved fits use a different selection rule; refit into a fresh --pool-dir")
    methods = dict(METHODS)
    if args.ablations:
        methods.update(ABLATIONS)
    labels = {"best_solo": "Solo reference", **methods}
    for judge in args.judges:
        labels[f"judge:{judge}:reasoning"] = f"Judge {judge}: reasoning" + (" (approx.)" if args.judge_policy == "approximate" else "")
        labels[f"judge:{judge}:reasoning_confidence"] = f"Judge {judge}: reasoning + confidence" + (" (approx.)" if args.judge_policy == "approximate" else "")
    judge_index = index_judges(args.judge_dir, args.judges)
    sources = SourceStore(args.store)
    rows, metadata = [], []
    order = {task: index for index, task in enumerate(args.tasks)}
    cells.sort(key=lambda cell: (order[cell["task"]], cell["models"], cell["estimator"]))
    for index, cell in enumerate(cells, 1):
        entries, info = process_cell(cell, sources, judge_index, args.judges, methods, args.reference, args.judge_policy)
        rows.extend(entries)
        metadata.append(info)
        print(f"[{index}/{len(cells)}] {cell['task']} {cell['estimator']} {cell['models']} "
              f"matched={info['n_matched']}/{info['n_evaluation']}", flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "atomic.csv", rows)
    (args.out_dir / "audit.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    files = write_tables(args.out_dir, rows, metadata, args.estimators, labels, args.tasks)
    statistics_path = args.out_dir / "significance.csv"
    if args.significance:
        tested_labels = {method: label for method, label in labels.items()
                         if not (args.judge_policy == "approximate" and method.startswith("judge:"))}
        tests = dataset_sign_tests(rows, metadata, args.estimators, tested_labels, args.tasks)
        write_csv(statistics_path, tests)
        files.extend(write_significance_tables(args.out_dir, tests, args.estimators, tested_labels))
    manifest = {"arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "pool_inputs": {str(args.pool_dir / f"{estimator}.csv"): digest(args.pool_dir / f"{estimator}.csv")
                                for estimator in args.estimators},
                "report_code": {str(path): digest(path) for path in sorted(Path(__file__).parent.glob("*.py"))},
                "inference_inputs": sources.paths, "n_panels": len(cells), "tables": files,
                "aggregation": "equal weight per model group within each dataset and size",
                "delta": "method minus panel-specific solo reference",
                "delta_spread": "sample standard deviation (ddof=1) across group deltas; descriptive, not a standard error or confidence interval", "display_units": "points (x100), except NLL in nats",
                "judge_score": "reused verbalized score; stale-target proxy when old and new answers differ" if args.judge_policy == "approximate" else "verbalized confidence in fixed answer",
                "judge_policy": args.judge_policy, "tie_break": args.tie_break, "tie_seed": args.tie_seed,
                "question_mask": "intersection of valid solo, pooling and all present selected judge scores per cell",
                "missing_judge_cells": "omitted; counts in separate coverage tables; no substitution",
                "inference": False, "refitting": args.refit}
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(files)} tables plus atomic.csv, audit.json and manifest.json to {args.out_dir}")
    print("Negative delta ECE and positive delta AUARC are improvements. Panel counts are in separate coverage tables.")


if __name__ == "__main__":
    main()
