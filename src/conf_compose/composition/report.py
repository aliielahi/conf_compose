"""Read saved panel runs and cut them: panel-size curve, matched budget, estimator family, pooling rule."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

FAMILY_ORDER = ["cons", "ver", "vertgt", "cons+ver", "cons+vertgt", "cons+ver+verb"]
SHORT = {"consistency": "cons", "verification": "ver", "verbalized": "verb", "seq": "seq",
         "ver_target": "vertgt"}


def load_runs(root: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Every saved panel run keyed by (task, target); the newest write wins for a repeated cell."""
    runs = {}
    for path in sorted(root.glob("*/panels_*/metrics.json"), key=lambda p: p.stat().st_mtime):
        task, target = path.parent.parts[-2], path.parent.name.split("_")[1]
        runs[(task, target)] = json.loads(path.read_text())
    return runs


def arm(row: Dict[str, Any]) -> str:
    """The three arms: one run's samples split, several runs of one model, several distinct models."""
    if row.get("kind") == "split":
        return "split"
    return "homo" if row.get("distinct_models", 0) <= 1 else "hetero"


def family_of(row: Dict[str, Any]) -> str:
    return "+".join(SHORT.get(name, name) for name in row["families"])


def pooled_rows(report: Dict[str, Any], rule: Optional[str] = None) -> Dict[str, Any]:
    """Only the panel rows, optionally one pooling rule; individual sources are excluded."""
    return {name: row for name, row in report.items()
            if "kind" in row and (rule is None or row.get("rule") == rule)}


def size_curve(report: Dict[str, Any], metric: str, rule: str) -> Dict[Tuple[str, str, int], Dict[int, list]]:
    """Metric against panel size, holding samples per stream fixed, per arm and family."""
    grid: Dict[Tuple[str, str, int], Dict[int, list]] = defaultdict(dict)
    for row in pooled_rows(report, rule).values():
        key = (arm(row), family_of(row), row["samples_per_stream"])
        size = row["models"] * row["streams_per_model"]
        grid[key].setdefault(size, []).append(row.get(metric))
    return grid


def matched_budget(report: Dict[str, Any], metric: str, rule: str) -> Dict[int, Dict[str, list]]:
    """Consistency-only panels grouped by sampled generations, so diversity is separated from sampling."""
    grid: Dict[int, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for row in pooled_rows(report, rule).values():
        if row["families"] == ["consistency"] and row.get(metric) is not None:
            grid[row["sample_calls"]][arm(row)].append(row[metric])
    return grid


def pick(report: Dict[str, Any], validation: Dict[str, Any], keep: Callable[[Dict[str, Any]], bool],
         metric: str) -> Tuple[Optional[str], Optional[float]]:
    """Panel chosen by validation AUROC, reported at test; never a maximum taken over the test column."""
    options = [(validation.get(name, {}).get("auroc"), name) for name, row in report.items()
               if "kind" in row and keep(row) and validation.get(name, {}).get("auroc") is not None]
    if not options:
        return None, None
    name = max(options)[1]
    return name, report[name].get(metric)


def single_sources(report: Dict[str, Any], metric: str) -> List[Tuple[float, str]]:
    return sorted(((row[metric], name) for name, row in report.items()
                   if name.startswith("single|") and row.get(metric) is not None), reverse=True)


def format_cell(report: Dict[str, Any], validation: Dict[str, Any], metric: str, rule: str,
                sizes: Sequence[int] = (2, 3, 4, 5)) -> List[str]:
    """The four cuts for one (task, target) cell, as printable lines."""
    lines = []
    lines += _table("best individual sources (test)", f"{'source':<28}{metric:>8}{'cov':>7}",
                    [f"{name.split('|')[1]:<28}{value:>8.3f}{report[name]['coverage']:>7.2f}"
                     for value, name in single_sources(report, metric)[:6]])

    curve = size_curve(report, metric, rule)
    rows = []
    for key in sorted(curve, key=lambda k: (k[1], k[0], k[2])):
        cells = "".join(f"{_mean(curve[key].get(size)):>9}" for size in sizes)
        rows.append(f"{key[0]:<7}{key[1]:<16}k={key[2]:<4}{cells}")
    lines += _table(f"panel size curve (S = {', '.join(map(str, sizes))})",
                    f"{'set':<7}{'families':<16}{'':<6}" + "".join(f"{'S=' + str(s):>9}" for s in sizes), rows)

    budget = matched_budget(report, metric, rule)
    lines += _table("consistency only, matched sampled generations",
                    f"{'samples':<10}{'hetero':>10}{'homo':>10}{'split':>10}",
                    [f"{calls:<10}{_mean(arms.get('hetero')):>10}{_mean(arms.get('homo')):>10}"
                     f"{_mean(arms.get('split')):>10}" for calls, arms in sorted(budget.items())])

    rows = []
    for family in FAMILY_ORDER:
        matches = [row for row in pooled_rows(report, rule).values()
                   if family_of(row) == family and arm(row) == "hetero"]
        values = [row.get(metric) for row in matches if row.get(metric) is not None]
        if not values:
            continue
        name, selected = pick(report, validation, lambda r, f=family: arm(r) == "hetero"
                              and family_of(r) == f and r.get("rule") == rule, metric)
        rows.append(f"{family:<16}{len(values):>5}{_cell(selected):>10}"
                    f"{sum(values) / len(values):>9.3f}  {_panel_of(name):<30}")
    lines += _table("estimator family, distinct-model panels (chosen on validation)",
                    f"{'family':<16}{'n':>5}{'val-sel':>10}{'mean':>9}  {'panel':<30}", rows)

    rows = []
    for name in ("mean", "logodds_sum", "logodds_mean"):
        pooled = [row for row in pooled_rows(report, name).values() if arm(row) == "hetero"]
        nlls = [row.get("nll") for row in pooled if row.get("nll") is not None]
        chosen, auroc_value = pick(report, validation,
                                   lambda r, u=name: arm(r) == "hetero" and r.get("rule") == u, "auroc")
        nll_value = report.get(chosen, {}).get("nll") if chosen else None
        rows.append(f"{name:<16}{len(pooled):>5}{_cell(auroc_value):>12}{_cell(nll_value):>10}{_mean(nlls):>10}")
    lines += _table("pooling rule, distinct-model panels (chosen on validation)",
                    f"{'rule':<16}{'n':>5}{'val auroc':>12}{'val nll':>10}{'mean nll':>10}", rows)
    return lines


def _table(title: str, header: str, rows: Sequence[str]) -> List[str]:
    return ["", f"### {title}", header, *rows]


def _mean(values) -> str:
    clean = [v for v in (values or []) if v is not None]
    return f"{sum(clean) / len(clean):.3f}" if clean else "—"


def _cell(value) -> str:
    return "—" if value is None else f"{value:.3f}"


def _panel_of(name: Optional[str]) -> str:
    return name.split("|")[0][:28] if name else "—"
