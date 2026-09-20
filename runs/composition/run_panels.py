"""Voting sweep: pool confidence over panel size, model set (homo/hetero) and estimator family."""

import argparse
import glob
import itertools
import hashlib
import json
from pathlib import Path

from conf_compose.composition import (Panel, Prediction, Row, available, base_model, candidate_set, evaluate,
                                      fit_intercepts, format_table, from_zero_shot, majority_answer,
                                      pool_methods, sources)
from conf_compose.constants import RESULTS_DIR, TASKS
from conf_compose.data import Example, get_task

FAMILIES = {"cons": ("consistency",), "ver": ("verification",), "cons+ver": ("consistency", "verification"),
            "cons+ver+verb": ("consistency", "verification", "verbalized"),
            "vertgt": ("ver_target",), "cons+vertgt": ("consistency", "ver_target")}
RULES = ("mean", "logodds_sum", "logodds_mean")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument("--validation", nargs="+", required=True)
    parser.add_argument("--test", nargs="+", required=True)
    parser.add_argument("--target", choices=["anchor", "majority"], default="majority")
    parser.add_argument("--sizes", type=int, nargs="+", default=[2, 3, 4, 5])
    parser.add_argument("--samples", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--families", nargs="+", default=list(FAMILIES), choices=list(FAMILIES))
    parser.add_argument("--target-scores", nargs="*", default=[],
                        help="target_verification_<model>_<split>.json files from runs/composition/score_targets.py")
    parser.add_argument("--variant", choices=["add_half", "epsilon"], default="add_half")
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--out-dir", default=str(RESULTS_DIR / "composition"))
    return parser.parse_args()


def voter_subsets(models, size):
    """Subsets of separately generated runs; past the distinct-model count, add one rotating repeat."""
    bases = {}
    for model in models:
        bases.setdefault(base_model(model), []).append(model)
    if size <= len(bases):
        return [subset for subset in itertools.combinations(models, size)
                if len({base_model(model) for model in subset}) == size]
    if size - len(bases) > 1:
        return []
    one_each = [runs[0] for runs in bases.values()]
    return [tuple(one_each + [runs[1]]) for runs in bases.values() if len(runs) > 1]


def build_panels(models, args):
    """Voter panels of separately generated runs against same-run sample splits, at matched samples."""
    panels = []
    for family in args.families:
        names = FAMILIES[family]
        for size in args.sizes:
            for per_stream in args.samples:
                for subset in voter_subsets(models, size):
                    tag = "+".join(_short(model) for model in subset)
                    panels.append(Panel(subset, 1, per_stream, names,
                                        label=f"voters[{tag}]_s{size}_k{per_stream}_{family}"))
                if "consistency" in names:
                    for model in models:
                        panels.append(Panel((model,), size, per_stream, names,
                                            label=f"split[{_short(model)}]_s{size}_k{per_stream}_{family}"))
    return panels


def _short(name: str) -> str:
    """Trim the run-directory name down to its model tag, keeping any family suffix."""
    model, _, rest = name.partition(":")
    model = model.replace("vllm__", "").split("_val")[0]
    return f"{model}:{rest}" if rest else model


def attach_target_scores(items, paths, mode):
    """Give each stream its verifier's rating of the shared target, keyed by the model that produced it."""
    by_model = {}
    for path in paths:
        payload = json.loads(Path(path).read_text())
        by_model[payload["model"]] = payload["scores"]
    attached = 0
    for item in items:
        for stream in item.streams:
            for model, scores in by_model.items():
                if _short(model).split(":")[0] in _short(stream.model):
                    entry = scores.get(item.example_id, {}).get(mode)
                    if entry:
                        stream.signals["verification_target"] = entry["verification"]
                        attached += 1
    return attached


def target_of(task, item, mode):
    streams = item.round_streams((0,))
    if mode == "anchor":
        anchor = next((s for s in streams if s.agent == 0), None)
        return anchor.answer if anchor else None
    return majority_answer(task, streams)


def collect(task, items, panels, args):
    """One prediction per (panel, rule) and per individual source, all rating the same fixed target."""
    predictions, sizes = {}, {}
    specs = {p.name: {"kind": p.kind, "models": len(p.models), "distinct_models": p.distinct,
                      "streams_per_model": p.streams_per_model, "samples_per_stream": p.samples_per_stream,
                      "families": list(p.families), "sample_calls": p.sample_calls,
                      "answer_calls": p.answer_calls, "signal_calls": p.signal_calls} for p in panels}
    full = _full_panel(items)
    for item in items:
        streams = item.round_streams((0,))
        target = target_of(task, item, args.target)
        if target is None:
            continue
        candidates = candidate_set(task, item, streams)
        if not any(task.equivalent(c, target) for c in candidates):
            candidates = [target, *candidates]
        correct = float(task.is_correct(target, Example(item.example_id, item.question, item.gold)))
        for panel in panels:
            panel_sources = available(sources(task, item, target, candidates, panel, streams, args.variant))
            if not panel_sources:
                continue
            sizes.setdefault(panel.name, []).append(len(panel_sources))
            for rule, prediction in pool_methods([s.score for s in panel_sources]).items():
                if rule in RULES:
                    key = f"{panel.name}|{rule}"
                    predictions.setdefault(key, []).append((item.example_id, prediction, correct, target))
        for source in available(sources(task, item, target, candidates, full, streams, args.variant)):
            predictions.setdefault(f"single|{_short(source.name)}", []).append(
                (item.example_id, Prediction(source.score), correct, target))
    for name, values in sizes.items():
        specs[name]["mean_sources"] = sum(values) / len(values)
    return predictions, specs


def rows_from(predictions, total, specs):
    rows = []
    for method, entries in predictions.items():
        scored = [(e, p, c, t) for e, p, c, t in entries if p.score is not None]
        panel = method.split("|")[0]
        rows.append(Row(method=method, example_ids=[e for e, _, _, _ in scored],
                        scores=[p.score for _, p, _, _ in scored], correct=[c for _, _, c, _ in scored],
                        is_probability=all(p.is_probability for _, p, _, _ in scored) if scored else True,
                        answers=[t for _, _, _, t in scored], total=total,
                        logits=[p.logit for _, p, _, _ in scored],
                        meta={"panel": panel, "rule": method.split("|")[1], **specs.get(panel, {})}))
    return rows


def main():
    args = parse_args()
    task = get_task(args.task)
    validation = from_zero_shot({Path(p).parent.name: Path(p) for p in sorted(_expand(args.validation))})
    test = from_zero_shot({Path(p).parent.name: Path(p) for p in sorted(_expand(args.test))})
    if args.target_scores:
        scores = _expand(args.target_scores)
        print(f"target scores: attached {attach_target_scores(validation, [p for p in scores if 'validation' in p], args.target)}"
              f" validation, {attach_target_scores(test, [p for p in scores if 'test' in p], args.target)} test")
    models = [s.model for s in sorted(validation[0].round_streams((0,)), key=lambda s: s.agent)]
    panels = build_panels(models, args)
    print(f"{args.task} target={args.target}: validation={len(validation)} test={len(test)} "
          f"models={len(models)} panels={len(panels)}")

    val_predictions, val_specs = collect(task, validation, panels, args)
    test_predictions, test_specs = collect(task, test, panels, args)
    val_rows = rows_from(val_predictions, len(validation), val_specs)
    val_report = evaluate(val_rows, reference="", n_boot=0)
    reference = max(((val_report[r.method].get("auroc") or 0.0, r.method)
                     for r in val_rows if r.method.startswith("single|")))[1]
    print(f"validation-selected reference = {reference}")

    intercepts = fit_intercepts(val_rows)
    test_rows = rows_from(test_predictions, len(test), test_specs)
    report = evaluate(test_rows, reference, args.n_boot, intercepts)
    print("\n".join(format_table(report, reference)))

    out_dir = Path(args.out_dir) / args.task / f"panels_{args.target}_{_run_id(args)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(
        {"reference": reference, "report": report, "validation": val_report}, indent=2))
    (out_dir / "predictions.json").write_text(json.dumps(
        {row.method: {"ids": row.example_ids, "scores": row.scores, "correct": row.correct} for row in test_rows}))
    (out_dir / "manifest.json").write_text(json.dumps(
        {"args": vars(args), "inputs": _hashes(args), "code": _code_hash(), "intercepts": intercepts,
         "models": models}, indent=2))
    print(f"\nsaved {out_dir}")


def _full_panel(items):
    models = [s.model for s in sorted(items[0].round_streams((0,)), key=lambda s: s.agent)]
    return Panel(tuple(models), 1, 5, ("consistency", "verification", "verbalized", "seq"))


def _expand(paths):
    expanded = []
    for path in paths:
        expanded += sorted(glob.glob(path)) or [path]
    return expanded


def _hashes(args):
    return {p: hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]
            for p in _expand(args.validation) + _expand(args.test)}


def _code_hash():
    """Hash of the composition sources, so a changed implementation cannot overwrite an old run."""
    root = Path(__file__).resolve().parents[2] / "src" / "conf_compose" / "composition"
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def _run_id(args):
    payload = json.dumps({"inputs": _hashes(args), "code": _code_hash(), "variant": args.variant,
                          "sizes": args.sizes, "samples": args.samples, "families": args.families},
                         sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:8]


if __name__ == "__main__":
    main()
