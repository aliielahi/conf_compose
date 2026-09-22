"""Report output-length percentiles per task, and the max_tokens each one needs to keep truncation rare."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from conf_compose.composition import base_model
from conf_compose.constants import TASKS
from conf_compose.pipelines.inference import STORE


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--budget", type=float, default=2.0, help="tolerated truncation percentage")
    parser.add_argument("--store", default=str(STORE))
    return parser.parse_args()


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))] if ordered else 0


def main():
    args = parse_args()
    keep = 1 - args.budget / 100
    for task in args.tasks:
        cells = sorted(Path(args.store, task).glob(f"*/{args.split}.jsonl"))
        if not cells:
            print(f"\n{task}: no cells under {args.store}")
            continue
        limit = TASKS[task]["max_tokens"]
        lengths, truncated = defaultdict(list), defaultdict(int)
        for path in cells:
            model = base_model(path.parent.name)
            for line in path.read_text().splitlines():
                row = json.loads(line)
                tokens = row.get("token_logprobs", {}).get("response")
                lengths[model].append(len(tokens) if tokens else 0)
                truncated[model] += row["finish_reason"] == "length"

        print(f"\n{task}: max_tokens={limit}, word_limit={TASKS[task]['word_limit']}, "
              f"tolerating {args.budget}% truncation")
        print(f"{'model':<20}{'n':>7}{'median':>8}{'p90':>7}{'p98':>7}{'p99.5':>8}{'trunc %':>9}{'needs':>8}")
        needed = 0
        for model, values in sorted(lengths.items()):
            share = 100 * truncated[model] / len(values)
            target = percentile(values, keep)
            needed = max(needed, target)
            print(f"{model.replace('vllm__', '')[:19]:<20}{len(values):>7}{percentile(values, 0.5):>8}"
                  f"{percentile(values, 0.9):>7}{percentile(values, 0.98):>7}{percentile(values, 0.995):>8}"
                  f"{share:>9.1f}{target:>8}")
        suggested = 256 * ((int(needed * 1.15) + 255) // 256)
        print(f"  suggested max_tokens {suggested} (worst model needs {needed}, plus 15% headroom); "
              f"word_limit about {int(suggested / 1.4)}")
        print(f"  cost multiplier vs now: roughly {suggested / limit:.1f}x for the models that truncate")


if __name__ == "__main__":
    main()
