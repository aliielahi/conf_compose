import argparse
import csv
import hashlib
import json
from argparse import Namespace
from pathlib import Path

import atomic
from conf_compose.constants import COMPOSITION_PANELS, RESULTS_DIR, ROOT, TASKS


def parse_args():
    parser = argparse.ArgumentParser(description="Compare learned pooling on the exact judge panels, without inference")
    atomic.add_common_arguments(parser)
    parser.set_defaults(match="_cs7s")
    parser.add_argument("--tasks", nargs="+", choices=sorted(TASKS),
                        default=["csqa", "boolq", "gsm8k", "truthfulqa", "gpqa"])
    parser.add_argument("--estimators", nargs="+", choices=("cons", "seq"), default=["cons", "seq"])
    parser.add_argument("--panel-sizes", nargs="+", type=int)
    parser.add_argument("--out-dir", default=str(RESULTS_DIR / "voting_atomic" / "learned_panels"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def code_digest():
    sources = sorted((ROOT / "src/conf_compose/composition").rglob("*.py"))
    sources += [ROOT / "src/conf_compose/utils/metrics.py", ROOT / "src/conf_compose/constants.json",
                Path(atomic.__file__), Path(__file__)]
    return hashlib.sha256(json.dumps([(str(path.relative_to(ROOT)), file_digest(path))
                                     for path in sources]).encode()).hexdigest()


def append_row(path, row):
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()
    if args.samples < 1 or not 0 < args.fit_fraction < 1:
        raise SystemExit("samples must be positive and fit-fraction must be between zero and one")
    args.tasks = list(dict.fromkeys(args.tasks))
    args.estimators = list(dict.fromkeys(args.estimators))
    panels = [panel for panel in COMPOSITION_PANELS if not args.panel_sizes or len(panel) in args.panel_sizes]
    if not panels:
        raise SystemExit("no judge panels have the requested sizes")
    if "seq" in args.estimators and not args.match:
        raise SystemExit("sequence scoring requires --match to select candidate-scored cells")
    for index, panel in enumerate(panels, 1):
        print(f"panel {index}: {', '.join(panel)}", flush=True)
    total = len(args.tasks) * len(args.estimators) * len(panels)
    print(f"{total} atomic rows; {args.samples} samples; majority answer; estimators={args.estimators}", flush=True)
    if args.dry_run:
        return
    out = Path(args.out_dir)
    outputs = {estimator: out / f"{estimator}.csv" for estimator in args.estimators}
    if any(path.exists() for path in outputs.values()):
        raise SystemExit("output files already exist; choose a fresh --out-dir to avoid mixing or duplicating runs")
    models = list(dict.fromkeys(f"vllm/{model}" for panel in panels for model in panel))
    args.group = models
    paths = {task: atomic.cell_paths(task, args, "test") for task in args.tasks}
    if args.fit_split == "validation":
        missing = [str(path.with_name("validation.jsonl")) for task_paths in paths.values()
                   for path in task_paths.values() if not path.with_name("validation.jsonl").is_file()]
        if missing:
            raise SystemExit(f"missing validation files: {missing}")
    out.mkdir(parents=True, exist_ok=True)
    code_hash = code_digest()
    completed = 0
    for task in args.tasks:
        records = atomic.load_records(paths[task])
        fitting_records = atomic.validation_records(paths[task]) if args.fit_split == "validation" else None
        input_hashes = {model: {"test": file_digest(path)} for model, path in paths[task].items()}
        if fitting_records is not None:
            for model, path in paths[task].items():
                input_hashes[model]["validation"] = file_digest(path.with_name("validation.jsonl"))
        for panel in panels:
            group = [f"vllm/{model}" for model in panel]
            items = atomic.group_items(records, group)
            fitting = atomic.group_items(fitting_records, group) if fitting_records is not None else None
            for estimator in args.estimators:
                case = Namespace(**{**vars(args), "group": group, "estimator": estimator})
                row = atomic.run_task(task, case, items, records, fitting, fitting_records)
                row["code_hash"] = code_hash
                row["input_hashes"] = json.dumps({model: input_hashes[model] for model in group}, sort_keys=True)
                append_row(outputs[estimator], row)
                completed += 1
                print(f"[{completed}/{total}] {task} {estimator} {','.join(panel)} "
                      f"coverage={row['vote_coverage']} blp={row['blp_status']} "
                      f"logistic={row['logistic_pool_status']}", flush=True)
    for path in outputs.values():
        print(path)


if __name__ == "__main__":
    main()
