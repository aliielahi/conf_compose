"""Independent-voting composition on round-0 evidence: fixed-answer confidence and selection, offline."""

import argparse
import hashlib
import json
from pathlib import Path

from conf_compose.composition import (Row, anchor_prior, evaluate, fit_intercepts, fixed_answer_methods,
                                      format_table, from_debate_traces, from_zero_shot, selection_methods)
from conf_compose.constants import RESULTS_DIR, TASKS
from conf_compose.data import Example, get_task
from conf_compose.debate import read_traces
from conf_compose.pipelines.runs import expand_globs, file_hashes, source_hash

FIXED_REFERENCE = "support_r0_a0"
SELECTION_REFERENCE = "linear_pool"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument("--validation", nargs="+", required=True, help="debate trace or zero-shot jsonl per model")
    parser.add_argument("--test", nargs="+", required=True)
    parser.add_argument("--source", choices=["debate", "zero_shot"], default="zero_shot")
    parser.add_argument("--budget", type=int, default=5, help="first K requested samples per stream")
    parser.add_argument("--variant", choices=["add_half", "epsilon"], default="add_half")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--out-dir", default=str(RESULTS_DIR / "composition"))
    return parser.parse_args()


def best_support_stream(rows) -> str:
    """Strongest individual sample-support stream by validation AUROC, chosen without touching test."""
    import numpy as np

    from conf_compose.utils.metrics import auroc
    options = []
    for row in rows:
        scores, correct = np.array(row.scores, dtype=float), np.array(row.correct, dtype=float)
        if row.method.startswith("support_r") and len(scores) > 2 and 0 < correct.sum() < len(correct):
            options.append((auroc(scores, correct), row.method))
    return max(options)[1] if options else FIXED_REFERENCE


def load_items(paths, source: str):
    if source == "debate":
        return from_debate_traces(read_traces(Path(paths[0])), rounds=(0,))
    return from_zero_shot({Path(p).parent.name: Path(p) for p in sorted(paths)})


def predictions(task, items, args, prior=None):
    """Family A (fixed anchor answer) and family B (selection), one prediction per example and method."""
    fixed, selection = {}, {}
    for item in items:
        streams = item.round_streams((0,))
        anchor = next((s for s in streams if s.agent == 0), None)
        example = Example(item.example_id, item.question, item.gold)
        if anchor and anchor.answer is not None:
            target_correct = float(task.is_correct(anchor.answer, example))
            for name, prediction in fixed_answer_methods(task, item, streams, anchor.answer, args.budget,
                                                         args.variant, anchor.tokens, prior).items():
                fixed.setdefault(name, []).append((item.example_id, prediction, target_correct))
        for name, prediction in selection_methods(task, item, streams, args.budget, args.variant).items():
            correct = float(task.is_correct(prediction.answer, example))
            selection.setdefault(name, []).append((item.example_id, prediction, correct))
    return fixed, selection


def rows_from(predictions_by_method, total: int):
    rows = []
    for method, entries in predictions_by_method.items():
        scored = [(example_id, prediction, correct) for example_id, prediction, correct in entries
                  if prediction.score is not None]
        rows.append(Row(method=method, example_ids=[e for e, _, _ in scored],
                        scores=[p.score for _, p, _ in scored], correct=[c for _, _, c in scored],
                        is_probability=all(p.is_probability for _, p, _ in scored) if scored else True,
                        answers=[p.answer for _, p, _ in scored], total=total))
    return rows


def main():
    args = parse_args()
    task = get_task(args.task)
    validation = load_items(expand_globs(args.validation), args.source)
    test = load_items(expand_globs(args.test), args.source)
    print(f"{args.task}: validation={len(validation)} test={len(test)} streams/example="
          f"{len(validation[0].round_streams((0,)))} budget={args.budget} smoothing={args.variant}")

    prior = anchor_prior(task, validation)
    val_fixed, val_selection = predictions(task, validation, args, prior)
    test_fixed, test_selection = predictions(task, test, args, prior)
    best_single = best_support_stream(rows_from(val_fixed, len(validation)))
    print(f"validation prior (anchor correctness) = {prior:.3f} | best single stream on validation = {best_single}")

    report = {"prior": prior, "best_single_stream": best_single}
    for family, reference, val, test_preds in (("fixed_answer", FIXED_REFERENCE, val_fixed, test_fixed),
                                               ("selection", SELECTION_REFERENCE, val_selection, test_selection)):
        intercepts = fit_intercepts(rows_from(val, len(validation)))
        extra = (best_single,) if family == "fixed_answer" else ()
        report[family] = evaluate(rows_from(test_preds, len(test)), reference, args.n_boot, intercepts, extra)
        print(f"\n=== {family} (ref *: {reference}; best = {best_single if extra else '—'}) ===")
        print("\n".join(format_table(report[family], reference, accuracy=family == "selection",
                                     second=best_single if extra else None)))

    out_dir = Path(args.out_dir) / args.task / _run_id(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    (out_dir / "manifest.json").write_text(json.dumps(
        {"args": vars(args), "inputs": _hashes(args), "code": source_hash("conf_compose.composition")}, indent=2))
    print(f"\nsaved {out_dir}")


def _hashes(args):
    return file_hashes(expand_globs(args.validation) + expand_globs(args.test))


def _run_id(args) -> str:
    payload = json.dumps({"inputs": _hashes(args), "budget": args.budget, "variant": args.variant,
                          "code": source_hash("conf_compose.composition")}, sort_keys=True)
    return f"{args.source}_k{args.budget}_{args.variant}_{hashlib.sha256(payload.encode()).hexdigest()[:8]}"


if __name__ == "__main__":
    main()
