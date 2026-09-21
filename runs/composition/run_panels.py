"""Voting sweep: pool confidence over panel size, model set (same/distinct) and estimator family."""

import argparse
import hashlib
import json
from pathlib import Path

from conf_compose.composition import (FAMILIES, GridConfig, attach_target_scores, best_by_validation,
                                      build_panels, collect, evaluate, fit_intercepts, format_table,
                                      from_zero_shot, rows_from)
from conf_compose.constants import RESULTS_DIR, TASKS
from conf_compose.data import get_task
from conf_compose.pipelines.runs import expand_globs, file_hashes, source_hash


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument("--validation", nargs="+", required=True)
    parser.add_argument("--test", nargs="+", required=True)
    parser.add_argument("--target", choices=["anchor", "majority"], default="majority")
    parser.add_argument("--sizes", type=int, nargs="+", default=[2, 3, 4, 5])
    parser.add_argument("--samples", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--families", nargs="+", default=list(FAMILIES), choices=list(FAMILIES))
    parser.add_argument("--target-scores", nargs="*", default=[])
    parser.add_argument("--cap", type=int, default=5, help="max panels per (size, family, budget) and arm")
    parser.add_argument("--variant", choices=["add_half", "epsilon"], default="add_half")
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--out-dir", default=str(RESULTS_DIR / "composition"))
    return parser.parse_args()


def load_split(paths):
    return from_zero_shot({Path(p).parent.name: Path(p) for p in sorted(expand_globs(paths))})


def main():
    args = parse_args()
    config = GridConfig(args.target, args.sizes, args.samples, args.families, args.variant, args.cap)
    task = get_task(args.task)
    validation, test = load_split(args.validation), load_split(args.test)
    if args.target_scores:
        scores = expand_globs(args.target_scores)
        print(f"target scores: attached "
              f"{attach_target_scores(validation, [p for p in scores if 'validation' in p], args.target)} validation, "
              f"{attach_target_scores(test, [p for p in scores if 'test' in p], args.target)} test")

    models = [s.model for s in sorted(validation[0].round_streams((0,)), key=lambda s: s.agent)]
    panels = build_panels(models, config)
    print(f"{args.task} target={args.target}: validation={len(validation)} test={len(test)} "
          f"models={len(models)} panels={len(panels)}", flush=True)

    val_rows = rows_from(*collect(task, validation, panels, config, progress=100), total=len(validation))
    val_report = evaluate(val_rows, reference="", n_boot=0)
    reference = best_by_validation(val_rows, val_report, "single|")
    print(f"validation-selected reference = {reference}", flush=True)

    test_rows = rows_from(*collect(task, test, panels, config, progress=100), total=len(test))
    print(f"evaluating {len(test_rows)} rows with n_boot={args.n_boot}", flush=True)
    intercepts = fit_intercepts(val_rows)
    report = evaluate(test_rows, reference, args.n_boot, intercepts)
    top = sorted(report, key=lambda m: -(report[m].get("auroc") or 0))[:15]
    print("\n".join(format_table({m: report[m] for m in top}, reference)))

    out_dir = Path(args.out_dir) / args.task / f"panels_{args.target}_{_run_id(args)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(
        {"reference": reference, "report": report, "validation": val_report}, indent=2))
    (out_dir / "predictions.json").write_text(json.dumps(
        {row.method: {"ids": row.example_ids, "scores": row.scores, "correct": row.correct} for row in test_rows}))
    (out_dir / "manifest.json").write_text(json.dumps(
        {"args": vars(args), "inputs": _inputs(args), "code": source_hash("conf_compose.composition"),
         "intercepts": intercepts, "models": models}, indent=2))
    print(f"\nsaved {out_dir}")


def _inputs(args):
    return file_hashes(expand_globs(args.validation) + expand_globs(args.test))


def _run_id(args):
    payload = json.dumps({"inputs": _inputs(args), "code": source_hash("conf_compose.composition"),
                          "variant": args.variant, "sizes": args.sizes, "samples": args.samples,
                          "families": args.families, "cap": args.cap}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:8]


if __name__ == "__main__":
    main()
