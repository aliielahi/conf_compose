"""Judge baseline: a named judge reads a named panel on one dataset and reports a final answer and confidence."""

import argparse
import json
from pathlib import Path

from conf_compose.constants import RESULTS_DIR, TASKS
from conf_compose.data import get_task
from conf_compose.judge_for_conf import CONFIDENCE_MODES, Judge, JudgeConfig, LEVELS, PanelEntry, PanelView
from conf_compose.pipelines.inference import STORE, InferenceSettings, load_model, records_path

NAME = "experiment04-judge_baseline"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument("--judge", required=True, help="the model that decides, e.g. vllm/g3-12i")
    parser.add_argument("--panel", nargs="+", required=True, help="the models whose answers it reads")
    parser.add_argument("--level", default="answer_reasoning", choices=list(LEVELS))
    parser.add_argument("--shown-confidence", default="consistency_t0.7",
                        help="which saved signal to show the judge at the confidence level")
    parser.add_argument("--modes", nargs="+", default=list(CONFIDENCE_MODES), choices=list(CONFIDENCE_MODES))
    parser.add_argument("--voter", type=int, default=0)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, help="first N examples, for a smoke run")
    parser.add_argument("--word-limit", type=int, default=150)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--store", default=str(STORE))
    parser.add_argument("--out-dir", default=str(RESULTS_DIR / NAME))
    return parser.parse_args()


def panel_records(args):
    """Each panel model's saved records for this task, keyed by model then example id."""
    per_model = {}
    for model in args.panel:
        settings = InferenceSettings(task=args.task, model=model, n_val=0, n_test=-1,
                                     answer_temperature=0.7, voter=args.voter,
                                     verification_context=True, debias=True)
        path = records_path(settings, args.split, args.store)
        if not path.exists():
            raise SystemExit(f"missing panel records:\n  {path}")
        per_model[model] = {row["id"]: row for row in _read(path)}
    return per_model


def build_views(args, per_model):
    shared = sorted(set.intersection(*(set(rows) for rows in per_model.values())), key=_sort_key)
    if args.limit:
        shared = shared[:args.limit]
    views = []
    for example_id in shared:
        first = per_model[args.panel[0]][example_id]
        entries = [PanelEntry(model=model, answer=per_model[model][example_id]["prediction"],
                              reasoning=per_model[model][example_id]["response"],
                              confidence=per_model[model][example_id]["confidence"].get(
                                  args.shown_confidence))
                   for model in args.panel]
        views.append(PanelView(example_id, first.get("question", ""), first["gold"], entries))
    return views


def main():
    args = parse_args()
    task = get_task(args.task)
    views = build_views(args, panel_records(args))
    print(f"{args.task}: judge={args.judge} panel={len(args.panel)} level={args.level} "
          f"examples={len(views)} modes={','.join(args.modes)}")

    config = JudgeConfig(level=args.level, shown_confidence=args.shown_confidence, modes=tuple(args.modes),
                         word_limit=args.word_limit, max_tokens=args.max_tokens)
    verdicts = Judge(load_model(args.judge), task, config).run(views)

    out_dir = Path(args.out_dir) / args.task / _cell(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "verdicts.jsonl").open("w") as handle:
        for view, verdict in zip(views, verdicts):
            row = verdict.to_dict()
            row["gold"] = view.gold
            row["correct"] = task.is_correct(verdict.answer, _Example(view))
            handle.write(json.dumps(row) + "\n")
    (out_dir / "manifest.json").write_text(json.dumps(
        {"experiment": NAME, "args": vars(args), "examples": len(views)}, indent=2))

    answered = sum(v.answer is not None for v in verdicts)
    correct = sum(task.is_correct(v.answer, _Example(w)) for w, v in zip(views, verdicts))
    print(f"answered {answered}/{len(verdicts)} | accuracy {correct / max(len(verdicts), 1):.3f}")
    for mode in args.modes:
        have = sum(getattr(v, mode) is not None for v in verdicts)
        print(f"  {mode}: {have}/{len(verdicts)} scored")
    print(f"saved {out_dir}")


class _Example:
    """Minimal stand-in for grading a verdict against the gold answer."""

    def __init__(self, view):
        self.id, self.question, self.answer = view.example_id, view.question, view.gold


def _cell(args):
    panel = "+".join(m.split("/")[-1] for m in args.panel)
    return f"{args.judge.split('/')[-1]}__{args.level}__{panel}__v{args.voter}"


def _read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _sort_key(example_id):
    tail = example_id.rsplit("-", 1)[-1]
    return (int(tail), example_id) if tail.isdigit() else (0, example_id)


if __name__ == "__main__":
    main()
