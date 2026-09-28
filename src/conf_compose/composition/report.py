"""Read saved panel runs and cut them: panel-size curve, matched budget, estimator family, pooling rule."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from conf_compose.data import Example
from conf_compose.utils.metrics import auarc, auroc, brier, ece, nll

from .candidates import candidate_set, support_of
from .methods import majority_answer, pool_methods

FAMILY_ORDER = ["cons", "ver", "verb", "seq", "vertgt", "cons+ver", "cons+vertgt", "cons+ver+verb",
                "cons+ver+verb+seq"]
RULE_ORDER = ["mean", "logodds_sum", "logodds_mean"]
MATRIX_METRICS = ["coverage", "accuracy", "auroc", "auarc", "ece", "brier", "nll"]
SHORT = {"consistency": "cons", "verification": "ver", "verbalized": "verb", "seq": "seq",
         "ver_target": "vertgt"}


def atomic_consistency_row(task, items, models: Sequence[str], samples: int = 5) -> Dict[str, Any]:
    """One exact panel: members rate their own answers; pooled scores rate this panel's vote.

    Accuracy uses every supplied question (missing answers are incorrect). Confidence metrics
    use available scores, with separate coverage. No parameters are fitted. Model/list order is
    preserved, and the existing majority rule breaks ties in agent order.
    """
    if not items or not models or len(set(models)) != len(models) or samples < 1:
        raise ValueError("atomic reporting needs examples, distinct models and a positive sample count")
    single_correct = [0] * len(models)
    single_answered = [0] * len(models)
    single_scores = [[] for _ in models]
    single_labels = [[] for _ in models]
    pooled_scores = {rule: [] for rule in RULE_ORDER}
    pooled_labels = []
    voting_correct = voting_answered = ties = full_sources = 0
    source_counts = []
    for item in items:
        by_model = {stream.model: stream for stream in item.round_streams((0,))}
        streams = [by_model[model] for model in models]
        if [s.model for s in sorted(streams, key=lambda s: s.agent)] != list(models):
            raise ValueError("model order must match agent order for reproducible vote ties")
        example = Example(item.example_id, item.question, item.gold)
        candidates = candidate_set(task, item, streams)
        supports = [support_of(task, s.samples[:samples], candidates)
                    if len(s.samples) >= samples else None for s in streams]
        for i, (stream, support) in enumerate(zip(streams, supports)):
            correct = int(task.is_correct(stream.answer, example))
            single_correct[i] += correct
            single_answered[i] += stream.answer is not None
            score = support.binary(stream.answer) if support else None
            if score is not None:
                single_scores[i].append(score)
                single_labels[i].append(correct)

        target = majority_answer(task, streams)
        if target is None:
            continue
        correct = int(task.is_correct(target, example))
        voting_correct += correct
        voting_answered += 1
        votes = [sum(s.answer is not None and task.equivalent(s.answer, c) for s in streams)
                 for c in candidates]
        ties += votes.count(max(votes)) > 1
        scores = [score for support in supports if support is not None
                  if (score := support.binary(target)) is not None]
        if scores:
            source_counts.append(len(scores))
            full_sources += len(scores) == len(models)
            pooled_labels.append(correct)
            for rule, prediction in pool_methods(scores).items():
                pooled_scores[rule].append(prediction.score)

    single = [_atomic_metrics(scores, labels) for scores, labels in zip(single_scores, single_labels)]
    pooled = {rule: _atomic_metrics(scores, pooled_labels) for rule, scores in pooled_scores.items()}
    count = len(items)
    return {
        "task": task.name, "n_models": len(models), "models": list(models),
        "samples_per_model": samples,
        "single_accuracy": [value / count for value in single_correct],
        "single_ece": [row["ece"] for row in single],
        "single_auarc": [row["auarc"] for row in single],
        "voting_accuracy": voting_correct / count,
        **{f"{rule}_ece_auarc": [pooled[rule]["ece"], pooled[rule]["auarc"]] for rule in RULE_ORDER},
        "n_eval": count,
        "single_answer_coverage": [value / count for value in single_answered],
        "single_confidence_coverage": [len(scores) / count for scores in single_scores],
        "voting_answer_coverage": voting_answered / count,
        "voting_confidence_coverage": len(pooled_labels) / count,
        "voting_full_source_coverage": full_sources / count,
        "voting_mean_sources": sum(source_counts) / len(source_counts) if source_counts else None,
        "vote_tie_rate": ties / count,
        "single_auroc": [row["auroc"] for row in single],
        "single_brier": [row["brier"] for row in single],
        "single_nll": [row["nll"] for row in single],
        "pooling_order": list(RULE_ORDER),
        **{f"pooling_{metric}": [pooled[rule][metric] for rule in RULE_ORDER]
           for metric in ("auroc", "brier", "nll")},
        "estimator": "consistency", "selection": "majority", "smoothing": "add_half",
        "calibration": "none", "tie_break": "first_model_in_cli_order",
    }


def _atomic_metrics(scores, labels):
    """Reliability metrics remain defined for one-class data; AUROC does not."""
    out = {}
    for name, metric in (("ece", ece), ("auarc", auarc), ("auroc", auroc), ("brier", brier), ("nll", nll)):
        value = metric(scores, labels) if scores else None
        out[name] = value if value is not None and math.isfinite(value) else None
    return out


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
    lines += family_matrix(report)
    lines += single_matrix(report)
    return lines


def single_matrix(report: Dict[str, Any], metrics: Sequence[str] = MATRIX_METRICS) -> List[str]:
    """One row per individual source rating the shared target: the baseline each panel must beat."""
    rows = []
    for name, row in sorted(report.items()):
        if not name.startswith("single|") or row.get("auroc") is None:
            continue
        rows.append((row["auroc"], f"{name.split('|')[1][:27]:<28}"
                     + "".join(f"{_cell(row.get(m)):>8}" for m in metrics)))
    header = f"{'source':<28}" + "".join(f"{m[:7]:>8}" for m in metrics)
    return _table("individual sources on the shared target (baselines)", header,
                  [line for _, line in sorted(rows, reverse=True)])


def family_matrix(report: Dict[str, Any], metrics: Sequence[str] = MATRIX_METRICS,
                  arms: Sequence[str] = ("hetero",)) -> List[str]:
    """Every estimator family x panel size x pooling rule, averaged over the panels of that shape."""
    cells: Dict[tuple, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for row in pooled_rows(report).values():
        if arm(row) not in arms or row.get("rule") not in RULE_ORDER:
            continue
        key = (family_of(row), row["models"] * row["streams_per_model"], row["rule"])
        for metric in metrics:
            if row.get(metric) is not None:
                cells[key][metric].append(row[metric])
        cells[key]["_n"].append(1)

    header = f"{'family':<20}{'S':>3}  {'rule':<14}{'n':>5}" + "".join(f"{m[:7]:>8}" for m in metrics)
    rows = []
    for family in FAMILY_ORDER:
        for size in sorted({key[1] for key in cells if key[0] == family}):
            for rule in RULE_ORDER:
                values = cells.get((family, size, rule))
                if not values:
                    continue
                rows.append(f"{family:<20}{size:>3}  {rule:<14}{len(values['_n']):>5}"
                            + "".join(f"{_mean(values.get(m)):>8}" for m in metrics))
    return _table("estimator family x panel size x pooling rule, mean over panels", header, rows)


def _table(title: str, header: str, rows: Sequence[str]) -> List[str]:
    return ["", f"### {title}", header, *rows]


def _mean(values) -> str:
    clean = [v for v in (values or []) if v is not None]
    return f"{sum(clean) / len(clean):.3f}" if clean else "—"


def _cell(value) -> str:
    return "—" if value is None else f"{value:.3f}"


def _panel_of(name: Optional[str]) -> str:
    return name.split("|")[0][:28] if name else "—"
