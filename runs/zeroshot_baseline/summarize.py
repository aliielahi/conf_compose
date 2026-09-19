"""Collect a sweep's report.json files into per-task markdown tables (summary.md) and a long CSV (summary.csv)."""

import argparse
import csv
import json
from pathlib import Path

from conf_compose.constants import BASELINES, RESULTS_DIR

METRICS = ("auroc", "cal_brier", "cal_ece", "raw_ece", "auarc")


def load_runs(out_dir: Path):
    for path in sorted(out_dir.glob("*/*/report.json")):
        payload = json.loads(path.read_text())
        yield payload["task"], payload["model"], payload


def markdown_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def fmt(value, ci=None):
    if value is None or value != value:
        return "–"
    return f"{value:.3f}" + (f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", default=BASELINES["sweep"])
    parser.add_argument("--signals", nargs="+", default=BASELINES["summary_signals"])
    args = parser.parse_args()
    out_dir = RESULTS_DIR / args.sweep

    runs = list(load_runs(out_dir))
    long_rows, sections = [], []
    for task in sorted({task for task, _, _ in runs}):
        task_runs = [(model, payload) for t, model, payload in runs if t == task]
        sections.append(f"## {task}")
        overview = [[model, str(p["splits"]["test"]["n"]), fmt(p["splits"]["test"]["accuracy"]),
                     str(p["splits"]["test"]["truncated"]), str(p["splits"]["test"]["no_answer"])]
                    for model, p in task_runs]
        sections.append(markdown_table(["model", "n_test", "accuracy", "truncated", "no_answer"], overview))

        for metric in ("auroc", "cal_brier", "cal_ece"):
            sections.append(f"### {metric}")
            rows = []
            for model, payload in task_runs:
                report = payload["report"]
                baseline = [fmt(report.get("constant", {}).get(metric))] if metric != "auroc" else []
                cells = [fmt(report.get(s, {}).get(metric), report.get(s, {}).get(f"{metric}_ci")) for s in args.signals]
                rows.append([model, *baseline, *cells])
            header = ["model", *(["constant"] if metric != "auroc" else []), *args.signals]
            sections.append(markdown_table(header, rows))

        for model, payload in task_runs:
            for name, row in payload["report"].items():
                long_rows += [{"task": task, "model": model, "signal": name, "metric": metric, "value": row.get(metric)}
                              for metric in METRICS if metric in row]

    markdown = "\n\n".join(sections) + "\n"
    (out_dir / "summary.md").write_text(markdown)
    print(markdown)
    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task", "model", "signal", "metric", "value"])
        writer.writeheader()
        writer.writerows(long_rows)
    print(f"saved {out_dir / 'summary.md'} and {csv_path}")


if __name__ == "__main__":
    main()
