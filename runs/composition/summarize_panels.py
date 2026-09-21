"""Summarize the voting-panel sweep: one block of cuts per saved (task, target) cell."""

import argparse
from pathlib import Path

from conf_compose.composition import format_cell, load_runs
from conf_compose.constants import RESULTS_DIR


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=str(RESULTS_DIR / "composition"))
    parser.add_argument("--metric", default="auroc", choices=["auroc", "auarc", "nll", "brier", "ece"])
    parser.add_argument("--rule", default="mean", choices=["mean", "logodds_sum", "logodds_mean"])
    return parser.parse_args()


def main():
    args = parse_args()
    runs = load_runs(Path(args.dir))
    if not runs:
        print(f"no panel runs under {args.dir}")
        return
    for (task, target), payload in sorted(runs.items()):
        print(f"\n{'=' * 78}\n{task} / target={target} / rule={args.rule} / metric={args.metric}"
              f"\nvalidation-selected single reference: {payload['reference']}")
        print("\n".join(format_cell(payload["report"], payload.get("validation", {}), args.metric, args.rule)))


if __name__ == "__main__":
    main()
