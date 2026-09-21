"""Move a legacy sweep directory into the shared inference store, recomputing each cell's digest."""

import argparse
import json
import re
import shutil
from pathlib import Path

from conf_compose.constants import TASKS
from conf_compose.pipelines.inference import STORE, InferenceSettings, inference_dir

LEGACY = re.compile(r"^(?P<model>.+?)(?:_rep(?P<voter>\d+))?_val(?P<val>\d+)_test(?P<test>\d+)$")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="e.g. results/sweep04")
    parser.add_argument("--answer-temperature", type=float, default=0.0)
    parser.add_argument("--consistency-samples", type=int, default=5)
    parser.add_argument("--no-verbalized", action="store_true")
    parser.add_argument("--store", default=str(STORE))
    parser.add_argument("--move", action="store_true", help="move instead of copy")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def settings_for(task, name, args):
    match = LEGACY.match(name)
    if not match or task not in TASKS:
        return None
    return InferenceSettings(task=task, model=match["model"].replace("__", "/", 1),
                             n_val=int(match["val"]), n_test=int(match["test"]),
                             answer_temperature=args.answer_temperature,
                             voter=int(match["voter"] or 0),
                             consistency_samples=args.consistency_samples,
                             verbalized=not args.no_verbalized).filled()


def main():
    args = parse_args()
    source = Path(args.source)
    moved = 0
    for run in sorted(source.glob("*/*")):
        if not run.is_dir():
            continue
        settings = settings_for(run.parent.name, run.name, args)
        if settings is None:
            print(f"skip {run} (unrecognised)")
            continue
        target = inference_dir(settings, args.store)
        print(f"{run}  ->  {target.relative_to(Path(args.store).parent)}")
        if args.dry_run:
            continue
        target.mkdir(parents=True, exist_ok=True)
        for split in ("validation", "test"):
            if (run / f"{split}.jsonl").exists():
                (shutil.move if args.move else shutil.copy2)(str(run / f"{split}.jsonl"),
                                                             str(target / f"{split}.jsonl"))
        legacy = json.loads((run / "report.json").read_text()) if (run / "report.json").exists() else {}
        (target / "settings.json").write_text(json.dumps(
            {"settings": settings.to_dict(), "migrated_from": str(run), "legacy": legacy}, indent=2))
        moved += 1

    for extra in sorted(source.glob("*/target_verification_*.json")):
        target = Path(args.store) / extra.parent.name / extra.name
        print(f"{extra}  ->  {target.relative_to(Path(args.store).parent)}")
        if not args.dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            (shutil.move if args.move else shutil.copy2)(str(extra), str(target))
    print(f"\n{moved} inference cell(s) {'moved' if args.move else 'copied'} into {args.store}")


if __name__ == "__main__":
    main()
