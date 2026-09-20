"""Summarize the voting-panel sweep: panel-size curve, homo vs hetero at matched budget, family and rule cuts."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from conf_compose.constants import RESULTS_DIR

FAMILY_ORDER = ["cons", "ver", "cons+ver", "cons+ver+verb"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=str(RESULTS_DIR / "composition"))
    parser.add_argument("--metric", default="auroc", choices=["auroc", "auarc", "nll", "brier", "ece"])
    parser.add_argument("--rule", default="mean", choices=["mean", "logodds_sum", "logodds_mean"])
    return parser.parse_args()


def load(root: Path):
    """Every saved panel run keyed by (task, target), newest wins on a repeated configuration."""
    runs = {}
    for path in sorted(root.glob("*/panels_*/metrics.json"), key=lambda p: p.stat().st_mtime):
        task, target = path.parent.parts[-2], path.parent.name.split("_")[1]
        runs[(task, target)] = json.loads(path.read_text())
    return runs


def rows_of(report, rule=None):
    return {name: row for name, row in report.items()
            if row.get("rule") and (rule is None or row["rule"] == rule)}


def size_curve(report, metric, rule):
    """Metric against panel size, holding samples per stream fixed, for each family and model set."""
    grid = defaultdict(dict)
    for name, row in rows_of(report, rule).items():
        key = (row["kind"], "+".join(_short(f) for f in row["families"]), row["samples_per_stream"])
        size = row["models"] * row["streams_per_model"]
        grid[key].setdefault(size, []).append(row.get(metric))
    return grid


def matched_budget(report, metric, rule):
    """Panels grouped by sampled generations consumed, so cross-model gain is separated from more sampling."""
    grid = defaultdict(lambda: defaultdict(list))
    for name, row in rows_of(report, rule).items():
        if row["families"] != ["consistency"] or row.get(metric) is None:
            continue
        grid[row["sample_calls"]][row["kind"]].append(row[metric])
    return grid


def pick(report, validation, keep, metric):
    """Panel chosen by validation AUROC, reported at test; never a maximum taken over the test column."""
    options = [(validation.get(name, {}).get("auroc"), name) for name, row in report.items()
               if row.get("rule") and keep(row) and validation.get(name, {}).get("auroc") is not None]
    if not options:
        return None, None
    name = max(options)[1]
    return name, report[name].get(metric)
    return grid


def table(title, header, lines):
    print(f"\n### {title}")
    print(header)
    print("\n".join(lines))


def main():
    args = parse_args()
    runs = load(Path(args.dir))
    if not runs:
        print(f"no panel runs under {args.dir}")
        return
    for (task, target), payload in sorted(runs.items()):
        report = payload["report"]
        validation = payload.get("validation", {})
        reference = payload["reference"]
        print(f"\n{'=' * 78}\n{task} / target={target} / rule={args.rule} / metric={args.metric}"
              f"\nvalidation-selected single reference: {reference}")

        singles = sorted(((row.get(args.metric), name) for name, row in report.items()
                          if name.startswith("single|") and row.get(args.metric) is not None), reverse=True)
        table("best individual sources (test)", f"{'source':<28}{args.metric:>8}{'cov':>7}",
              [f"{name.split('|')[1]:<28}{value:>8.3f}{report[name]['coverage']:>7.2f}"
               for value, name in singles[:6]])

        curve = size_curve(report, args.metric, args.rule)
        lines = []
        for (kind, families, per_stream) in sorted(curve, key=lambda k: (k[1], k[0], k[2])):
            sizes = curve[(kind, families, per_stream)]
            cells = "".join(f"{_mean(sizes.get(s)):>9}" for s in (2, 3, 4, 5))
            lines.append(f"{kind:<7}{families:<16}k={per_stream:<4}{cells}")
        table("panel size curve (S = 2,3,4,5)", f"{'set':<7}{'families':<16}{'':<6}"
              + "".join(f"{'S=' + str(s):>9}" for s in (2, 3, 4, 5)), lines)

        budget = matched_budget(report, args.metric, args.rule)
        lines = [f"{calls:<10}{_mean(kinds.get('voters')):>10}{_mean(kinds.get('split')):>10}"
                 for calls, kinds in sorted(budget.items())]
        table("consistency only, matched sampled generations",
              f"{'samples':<10}{'voters':>10}{'split':>10}", lines)

        lines = []
        for family in FAMILY_ORDER:
            matches = [row for row in rows_of(report, args.rule).values()
                       if "+".join(_short(f) for f in row["families"]) == family and row.get("kind") == "voters"]
            values = [row.get(args.metric) for row in matches if row.get(args.metric) is not None]
            if not values:
                continue
            name, selected = pick(report, validation, lambda r, f=family: r.get("kind") == "voters"
                                  and "+".join(_short(x) for x in r["families"]) == f and r["rule"] == args.rule,
                                  args.metric)
            lines.append(f"{family:<16}{len(values):>5}{_cell(selected):>10}"
                         f"{sum(values) / len(values):>9.3f}  {_panel_of(name):<30}")
        table("estimator family, voter panels (panel chosen on validation)",
              f"{'family':<16}{'n':>5}{'val-sel':>10}{'mean':>9}  {'panel':<30}", lines)

        lines = []
        for rule in ("mean", "logodds_sum", "logodds_mean"):
            rows = [row for row in rows_of(report, rule).values() if row.get("kind") == "voters"]
            nlls = [row.get("nll") for row in rows if row.get("nll") is not None]
            name, auroc_value = pick(report, validation,
                                     lambda r, u=rule: r.get("kind") == "voters" and r["rule"] == u, "auroc")
            nll_value = report.get(name, {}).get("nll") if name else None
            lines.append(f"{rule:<16}{len(rows):>5}{_cell(auroc_value):>12}{_cell(nll_value):>10}"
                         f"{_mean(nlls):>10}")
        table("pooling rule, voter panels (panel chosen on validation)",
              f"{'rule':<16}{'n':>5}{'val auroc':>12}{'val nll':>10}{'mean nll':>10}", lines)


def _mean(values):
    if not values:
        return "—"
    clean = [v for v in values if v is not None]
    return f"{sum(clean) / len(clean):.3f}" if clean else "—"


def _cell(value):
    return "—" if value is None else f"{value:.3f}"


def _panel_of(name):
    return name.split("|")[0][:28] if name else "—"


def _short(family):
    return {"consistency": "cons", "verification": "ver", "verbalized": "verb", "seq": "seq"}.get(family, family)


if __name__ == "__main__":
    main()
