"""Recompute a sweep's reports from saved records (re-parses verbalized ratings); no model inference."""

import argparse
import json
import statistics
from pathlib import Path

from conf_compose.confidence_estimators import parse_confidence
from conf_compose.constants import RESULTS_DIR
from conf_compose.pipelines.report import confidence_report
from conf_compose.pipelines.runs import split_summary


def reparse_verbalized(records):
    for record in records:
        if "verbalized_raw" in record:
            scores = [s for s in (parse_confidence(text) for text in record["verbalized_raw"]) if s is not None]
            record["confidence"]["verbalized"] = statistics.mean(scores) if scores else None


def load(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", required=True)
    args = parser.parse_args()

    for report_path in sorted((RESULTS_DIR / args.sweep).glob("*/*/report.json")):
        run = report_path.parent
        payload = json.loads(report_path.read_text())
        records = {split: load(run / f"{split}.jsonl") for split in ("validation", "test")}
        records = {split: rows for split, rows in records.items() if rows}
        for split, rows in records.items():
            reparse_verbalized(rows)
            (run / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        run_args = payload["args"]
        payload["splits"] = {split: split_summary(rows) for split, rows in records.items()}
        payload["report"] = confidence_report(records.get("validation", []), records["test"], run_args["calibrator"],
                                              run_args["n_boot"], answered_only=not run_args["include_unanswered"])
        report_path.write_text(json.dumps(payload, indent=2))
        print(f"updated {run}")


if __name__ == "__main__":
    main()
