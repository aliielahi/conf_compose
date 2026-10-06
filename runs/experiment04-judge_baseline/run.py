"""Judge as a confidence combiner: given a panel and its majority answer, report a confidence in that answer."""

import argparse
import json
from pathlib import Path

from conf_compose.constants import RESULTS_DIR, TASKS
from conf_compose.data import Example, get_task
from conf_compose.judge_for_conf import CONFIDENCE_MODES, Judge, JudgeConfig, PanelEntry, PanelView, VIEWS
from conf_compose.composition import base_model
from conf_compose.pipelines.inference import STORE, load_model
from conf_compose.utils.metrics import auarc, auroc, ece

NAME = "experiment04-judge_baseline"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument("--judge", required=True, help="the model reporting the confidence, e.g. vllm/g3-12i")
    parser.add_argument("--panel", nargs="+", required=True, help="the models whose evidence it reads")
    parser.add_argument("--view", default="reasoning_confidence", choices=list(VIEWS))
    parser.add_argument("--confidence-method", default="consistency_t0.7",
                        help="which saved signal the judge is shown, e.g. consistency_t0.7, verification, "
                             "verbalized, seq_response")
    parser.add_argument("--modes", nargs="+", default=list(CONFIDENCE_MODES), choices=list(CONFIDENCE_MODES))
    parser.add_argument("--voter", type=int, default=0)
    parser.add_argument("--match", default="", help="substring picking one cell per model, e.g. _cs7s")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, help="first N examples, for a smoke run")
    parser.add_argument("--word-limit", type=int, default=150)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--store", default=str(STORE))
    parser.add_argument("--out-dir", default=str(RESULTS_DIR / NAME))
    return parser.parse_args()


def panel_records(args):
    """One cell per panel model, resolved by name so a digest change cannot silently pick another run."""
    root = Path(args.store, args.task)
    per_model = {}
    for model in args.panel:
        cells = [p for p in sorted(root.glob(f"*/{args.split}.jsonl"))
                 if base_model(p.parent.name) == model.replace("/", "__")
                 and f"v{args.voter}_" in p.parent.name.split("--")[1]
                 and (not args.match or args.match in p.parent.name)]
        if not cells:
            raise SystemExit(f"no {args.split} cell for {model} at voter {args.voter} in {root}")
        if len(cells) > 1:
            listing = "\n".join(f"  {p.parent.name}" for p in cells)
            raise SystemExit(f"several cells for {model}; pass --match to pick one:\n{listing}")
        per_model[model] = ({row["id"]: row for row in _read(cells[0])}, cells[0].parent.name)
    return per_model


def majority(task, answers):
    """The aggregation the judge is handed; deciding it is not the judge's job here."""
    candidates = []
    for answer in answers:
        if answer is not None and not any(task.equivalent(c, answer) for c in candidates):
            candidates.append(answer)
    if not candidates:
        return None
    counts = {c: sum(task.equivalent(a, c) for a in answers if a is not None) for c in candidates}
    return max(candidates, key=lambda c: (counts[c], -candidates.index(c)))


def build_views(args, task, per_model):
    """Only examples every panel model answered, so each judgement sees the whole panel."""
    if args.view == "reasoning_confidence":
        for model, (rows, cell) in per_model.items():
            available = sorted(next(iter(rows.values()))["confidence"])
            if args.confidence_method not in available:
                raise SystemExit(f"{model} ({cell}) has no '{args.confidence_method}'; "
                                 f"available:\n  " + "\n  ".join(available))
    shared = sorted(set.intersection(*(set(rows) for rows, _ in per_model.values())), key=_sort_key)
    if args.limit:
        shared = shared[:args.limit]
    views = []
    for example_id in shared:
        rows = {model: per_model[model][0][example_id] for model in args.panel}
        entries = [PanelEntry(model=model, answer=row["prediction"], reasoning=row["response"],
                              confidence=row["confidence"].get(args.confidence_method))
                   for model, row in rows.items()]
        first = rows[args.panel[0]]
        views.append(PanelView(example_id, first.get("question", ""), first["gold"], entries,
                               majority(task, [e.answer for e in entries])))
    return views


def report(task, views, verdicts, modes):
    """Does the judge's confidence rank the majority answer's correctness?"""
    labels = [float(task.is_correct(v.final_answer, Example(w.example_id, w.question, w.gold)))
              for w, v in zip(views, verdicts)]
    written = sum(bool(v.justification) for v in verdicts)
    print(f"\nmajority accuracy {sum(labels) / max(len(labels), 1):.3f} over {len(labels)} example(s)")
    print(f"justifications written {written}/{len(verdicts)} (saved only, nothing reads them yet)")
    print(f"{'mode':<12}{'cov':>7}{'auroc':>8}{'auarc':>8}{'ece':>8}{'mean':>8}")
    for mode in modes:
        pairs = [(getattr(v, mode), y) for v, y in zip(verdicts, labels) if getattr(v, mode) is not None]
        if len(pairs) < 10 or not 0 < sum(y for _, y in pairs) < len(pairs):
            print(f"{mode:<12}{len(pairs) / max(len(verdicts), 1):>7.3f}{'—':>8}{'—':>8}{'—':>8}{'—':>8}")
            continue
        scores, ys = [s for s, _ in pairs], [y for _, y in pairs]
        print(f"{mode:<12}{len(pairs) / len(verdicts):>7.3f}{auroc(scores, ys):>8.3f}"
              f"{auarc(scores, ys):>8.3f}{ece(scores, ys):>8.3f}{sum(scores) / len(scores):>8.3f}")


def main():
    args = parse_args()
    task = get_task(args.task)
    per_model = panel_records(args)
    views = build_views(args, task, per_model)
    shown = args.confidence_method if args.view == "reasoning_confidence" else "none"
    print(f"{args.task}: judge={args.judge} view={args.view} shown_confidence={shown} "
          f"examples={len(views)} modes={','.join(args.modes)}")
    for model, (_, cell) in per_model.items():
        print(f"  panel {model:<20}{cell}")

    config = JudgeConfig(view=args.view, confidence_method=args.confidence_method,
                         modes=tuple(args.modes), word_limit=args.word_limit, max_tokens=args.max_tokens)
    verdicts = Judge(load_model(args.judge), task, config).run(views)

    out_dir = Path(args.out_dir) / args.task / _cell(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "verdicts.jsonl").open("w") as handle:
        for view, verdict in zip(views, verdicts):
            row = verdict.to_dict()
            row["gold"] = view.gold
            row["correct"] = task.is_correct(verdict.final_answer,
                                             Example(view.example_id, view.question, view.gold))
            handle.write(json.dumps(row) + "\n")
    (out_dir / "manifest.json").write_text(json.dumps(
        {"experiment": NAME, "args": vars(args), "examples": len(views)}, indent=2))

    report(task, views, verdicts, args.modes)
    print(f"\nsaved {out_dir}")


def _cell(args):
    panel = "+".join(m.split("/")[-1] for m in args.panel)
    method = args.confidence_method if args.view == "reasoning_confidence" else "none"
    return f"{args.judge.split('/')[-1]}__{args.view}__{method}__{panel}__v{args.voter}"


def _read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _sort_key(example_id):
    tail = example_id.rsplit("-", 1)[-1]
    return (int(tail), example_id) if tail.isdigit() else (0, example_id)


if __name__ == "__main__":
    main()
