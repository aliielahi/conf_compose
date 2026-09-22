"""Audit a task's inference cells: completeness, signal coverage, and whether the voters really differ."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from conf_compose.composition import base_model
from conf_compose.data import get_task
from conf_compose.pipelines.inference import STORE

SIGNALS = ["seq_response", "seq_response_min", "seq_response_tail10", "seq_response_debiased",
           "verification", "verification_context", "verbalized", "consistency_t0.7"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--expect", type=int, default=0, help="cells expected, 0 to skip the check")
    parser.add_argument("--store", default=str(STORE))
    return parser.parse_args()


def main():
    args = parse_args()
    task = get_task(args.task)
    cells = sorted(Path(args.store, args.task).glob(f"*/{args.split}.jsonl"))
    print(f"{args.task}/{args.split}: {len(cells)} cell(s)"
          + (f" (expected {args.expect})" if args.expect else ""))
    if not cells:
        return

    problems, by_model = [], defaultdict(dict)
    print(f"\n{'cell':<44}{'rows':>6}{'acc':>7}{'noans':>7}{'trunc':>7}{'resp':>6}{'missing signals'}")
    for path in cells:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        schema = json.loads((path.parent / "settings.json").read_text()).get("record_schema")
        missing = [name for name in SIGNALS
                   if sum(row["confidence"].get(name) is not None for row in rows) < 0.5 * len(rows)]
        responses = min((len(row.get("sampled_responses", {}).get("consistency_t0.7", [])) for row in rows),
                        default=0)
        accuracy = sum(row["correct"] for row in rows) / len(rows)
        print(f"{path.parent.name[:43]:<44}{len(rows):>6}{accuracy:>7.3f}"
              f"{sum(r['prediction'] is None for r in rows):>7}"
              f"{sum(r['finish_reason'] == 'length' for r in rows):>7}{responses:>6}  {','.join(missing) or '-'}")
        if schema != 2:
            problems.append(f"{path.parent.name}: record_schema={schema}, expected 2")
        if responses < 5:
            problems.append(f"{path.parent.name}: only {responses} saved resample reasoning(s)")
        for row in rows:
            by_model[_panel(path.parent.name)].setdefault(row["id"], []).append(row["prediction"])

    print(f"\n{'model':<28}{'voters':>7}{'all agree':>11}{'identical pairs':>17}")
    for model, answers in sorted(by_model.items()):
        counts = {len(v) for v in answers.values()}
        voters = max(counts)
        shared = [v for v in answers.values() if len(v) == voters]
        agree = sum(len({str(a) for a in v}) == 1 for v in shared) / len(shared) if shared else 0
        identical = _identical_voter_pairs(shared, voters)
        print(f"{model[:27]:<28}{voters:>7}{agree:>11.3f}{identical:>17}")
        if voters > 1 and identical:
            problems.append(f"{model}: {identical} voter pair(s) produced identical answers on every example")

    print("\n" + ("\n".join(f"PROBLEM  {p}" for p in problems) if problems else "no problems found"))


def _panel(cell: str) -> str:
    """Voters are comparable only within one model at one split size; a smaller cell is a different panel."""
    size = next((part for part in cell.split("--")[1].split("_") if part.startswith("n") or part == "full"), "")
    return f"{base_model(cell)} [{size}]"


def _identical_voter_pairs(shared, voters):
    """Two voters that never differ anywhere mean their sampling identities collided."""
    pairs = 0
    for i in range(voters):
        for j in range(i + 1, voters):
            if all(str(v[i]) == str(v[j]) for v in shared):
                pairs += 1
    return pairs


if __name__ == "__main__":
    main()
