"""Panel grids over a voter sweep: which runs form a panel, and one prediction per panel and rule."""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from conf_compose.data import Example

from .candidates import candidate_set
from .evaluate import Row
from .evidence import Item
from .methods import Prediction, majority_answer, pool_methods
from .panels import Panel, available, base_model, sources

FAMILIES = {"cons": ("consistency",), "ver": ("verification",), "cons+ver": ("consistency", "verification"),
            "cons+ver+verb": ("consistency", "verification", "verbalized"),
            "vertgt": ("ver_target",), "cons+vertgt": ("consistency", "ver_target")}
RULES = ("mean", "logodds_sum", "logodds_mean")


@dataclass
class GridConfig:
    """One sweep cell: the fixed target and the panel dimensions varied around it."""
    target: str = "majority"
    sizes: Sequence[int] = (2, 3, 4, 5)
    samples: Sequence[int] = (1, 2, 3, 4, 5)
    families: Sequence[str] = tuple(FAMILIES)
    variant: str = "add_half"
    cap: int = 5


def short_name(name: str) -> str:
    """Trim a run directory down to its model tag, keeping any family suffix after a colon."""
    model, _, rest = name.partition(":")
    model = model.replace("vllm__", "").split("_val")[0]
    return f"{model}:{rest}" if rest else model


def model_key(name: str) -> str:
    """One key for a model however it is written: a cli spec, a run directory, or a voter repeat."""
    return base_model(name.replace("/", "__").split("_val")[0])


def group_runs(models: Sequence[str]) -> Dict[str, List[str]]:
    """The separately generated runs of each base model, in voter order."""
    bases: Dict[str, List[str]] = {}
    for model in models:
        bases.setdefault(base_model(model), []).append(model)
    return {base: sorted(runs) for base, runs in bases.items()}


def hetero_subsets(models: Sequence[str], size: int, cap: int) -> List[Tuple[str, ...]]:
    """Panels of `size` distinct models, one run each; voter assignments rotate and are capped."""
    bases = group_runs(models)
    out = []
    for chosen in itertools.combinations(sorted(bases), min(size, len(bases))):
        if len(chosen) < min(size, len(bases)):
            continue
        for shift in range(min(cap, max(len(bases[base]) for base in chosen))):
            subset = [bases[base][(shift + i) % len(bases[base])] for i, base in enumerate(chosen)]
            if size > len(chosen):
                extra = bases[chosen[shift % len(chosen)]]
                if len(extra) < 2:
                    continue
                subset.append(extra[(shift + 1) % len(extra)])
            if len(subset) == size and len(set(subset)) == size:
                out.append(tuple(subset))
    return out


def homo_subsets(models: Sequence[str], size: int, cap: int) -> List[Tuple[str, ...]]:
    """Panels of `size` separate runs of one model: the arm a voter sweep is generated for."""
    out = []
    for _, runs in sorted(group_runs(models).items()):
        if len(runs) >= size:
            out += list(itertools.combinations(runs, size))[:cap]
    return out


def split_runs(models: Sequence[str], cap: int) -> List[str]:
    """One run per base model first, so the sample-split arm spans models instead of repeating one."""
    grouped = group_runs(models)
    chosen = [runs[0] for _, runs in sorted(grouped.items())]
    for depth in range(1, max(len(runs) for runs in grouped.values())):
        chosen += [runs[depth] for _, runs in sorted(grouped.items()) if len(runs) > depth]
    return chosen[:max(cap, len(grouped))]


def build_panels(models: Sequence[str], config: GridConfig) -> List[Panel]:
    """Same-model voter panels against distinct-model panels, plus the same-run sample split control."""
    panels = []
    for family in config.families:
        names = FAMILIES[family]
        for size in config.sizes:
            for per_stream in config.samples:
                for subset in hetero_subsets(models, size, config.cap):
                    panels.append(Panel(subset, 1, per_stream, names,
                                        label=f"hetero[{_tag(subset)}]_s{size}_k{per_stream}_{family}"))
                for subset in homo_subsets(models, size, config.cap):
                    panels.append(Panel(subset, 1, per_stream, names,
                                        label=f"homo[{_tag(subset)}]_s{size}_k{per_stream}_{family}"))
                if "consistency" in names:
                    for model in split_runs(models, config.cap):
                        panels.append(Panel((model,), size, per_stream, names,
                                            label=f"split[{short_name(model)}]_s{size}_k{per_stream}_{family}"))
    return panels


def attach_target_scores(items: Sequence[Item], paths: Sequence[str], mode: str) -> int:
    """Give each stream its verifier's rating of the shared target, keyed by the model that produced it."""
    by_model = {}
    for path in paths:
        payload = json.loads(Path(path).read_text())
        by_model[model_key(payload["model"])] = payload["scores"]
    attached = 0
    for item in items:
        for stream in item.streams:
            scores = by_model.get(model_key(stream.model))
            entry = scores.get(item.example_id, {}).get(mode) if scores else None
            if entry and entry.get("verification") is not None:
                stream.signals["verification_target"] = entry["verification"]
                attached += 1
    return attached


def target_of(task, item: Item, mode: str) -> Optional[str]:
    """The one answer every method in this cell must rate: an anchor's, or the panel's majority vote."""
    streams = item.round_streams((0,))
    if mode == "anchor":
        anchor = next((s for s in streams if s.agent == 0), None)
        return anchor.answer if anchor else None
    return majority_answer(task, streams)


def collect(task, items: Sequence[Item], panels: Sequence[Panel], config: GridConfig,
            progress: int = 0) -> Tuple[Dict[str, list], Dict[str, dict]]:
    """One prediction per (panel, rule) and per individual source, all rating the same fixed target."""
    predictions: Dict[str, list] = {}
    counts: Dict[str, list] = {}
    specs = {p.name: {"kind": p.kind, "models": len(p.models), "distinct_models": p.distinct,
                      "streams_per_model": p.streams_per_model, "samples_per_stream": p.samples_per_stream,
                      "families": list(p.families), "sample_calls": p.sample_calls,
                      "answer_calls": p.answer_calls, "signal_calls": p.signal_calls} for p in panels}
    full = full_panel(items)
    for done, item in enumerate(items, 1):
        if progress and done % progress == 0:
            print(f"  scored {done}/{len(items)} examples", flush=True)
        streams = item.round_streams((0,))
        target = target_of(task, item, config.target)
        if target is None:
            continue
        candidates = candidate_set(task, item, streams)
        if not any(task.equivalent(c, target) for c in candidates):
            candidates = [target, *candidates]
        correct = float(task.is_correct(target, Example(item.example_id, item.question, item.gold)))
        for panel in panels:
            panel_sources = available(sources(task, item, target, candidates, panel, streams, config.variant))
            if not panel_sources:
                continue
            counts.setdefault(panel.name, []).append(len(panel_sources))
            for rule, prediction in pool_methods([s.score for s in panel_sources]).items():
                if rule in RULES:
                    predictions.setdefault(f"{panel.name}|{rule}", []).append(
                        (item.example_id, prediction, correct, target))
        for source in available(sources(task, item, target, candidates, full, streams, config.variant)):
            predictions.setdefault(f"single|{short_name(source.name)}", []).append(
                (item.example_id, Prediction(source.score), correct, target))
    for name, values in counts.items():
        specs[name]["mean_sources"] = sum(values) / len(values)
    return predictions, specs


def rows_from(predictions: Dict[str, list], specs: Dict[str, dict], total: int) -> List[Row]:
    rows = []
    for method, entries in predictions.items():
        scored = [(e, p, c, t) for e, p, c, t in entries if p.score is not None]
        panel = method.split("|")[0]
        rows.append(Row(method=method, example_ids=[e for e, _, _, _ in scored],
                        scores=[p.score for _, p, _, _ in scored], correct=[c for _, _, c, _ in scored],
                        is_probability=all(p.is_probability for _, p, _, _ in scored) if scored else True,
                        answers=[t for _, _, _, t in scored], total=total,
                        logits=[p.logit for _, p, _, _ in scored],
                        meta={"panel": panel, "rule": method.split("|")[1], **specs.get(panel, {})}))
    return rows


def voter_rows(paths: Sequence[str]) -> List[Tuple[str, List[dict]]]:
    """Every voter's record per example id, over the ids all voters answered."""
    per_voter = [{row["id"]: row for row in _read_jsonl(path)} for path in paths]
    ids = sorted(set.intersection(*(set(rows) for rows in per_voter)), key=_example_order)
    return [(example_id, [rows[example_id] for rows in per_voter]) for example_id in ids]


def shared_targets(task, mode: str, rows_by_example: Sequence[Tuple[str, List[dict]]]) -> Dict[str, str]:
    """One target answer per example: the first voter's, or the ordinary majority vote across voters."""
    out = {}
    for example_id, rows in rows_by_example:
        answers = [row["prediction"] for row in rows if row["prediction"] is not None]
        if answers:
            out[example_id] = answers[0] if mode == "anchor" else _vote(task, answers)
    return out


def anchor_prior(task, items: Sequence[Item]) -> float:
    """Laplace-smoothed validation correctness of the anchor's answer; a reference rate, not a model prior."""
    correct = 0
    for item in items:
        anchor = next((s for s in item.round_streams((0,)) if s.agent == 0), None)
        if anchor and anchor.answer is not None:
            correct += task.is_correct(anchor.answer, Example(item.example_id, item.question, item.gold))
    return (correct + 1) / (len(items) + 2)


def full_panel(items: Sequence[Item]) -> Panel:
    """Every model once at full sample budget, used to score each individual source on its own."""
    models = [s.model for s in sorted(items[0].round_streams((0,)), key=lambda s: s.agent)]
    return Panel(tuple(models), 1, 5, ("consistency", "verification", "verbalized", "seq", "ver_target"))


def _tag(subset: Sequence[str]) -> str:
    return "+".join(short_name(model) for model in subset)


def _vote(task, answers: Sequence[str]) -> str:
    candidates: List[str] = []
    for answer in answers:
        if not any(task.equivalent(existing, answer) for existing in candidates):
            candidates.append(answer)
    counts = {c: sum(task.equivalent(a, c) for a in answers) for c in candidates}
    return max(candidates, key=lambda c: (counts[c], -candidates.index(c)))


def _read_jsonl(path) -> List[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _example_order(example_id: str):
    tail = example_id.rsplit("-", 1)[-1]
    return (int(tail), example_id) if tail.isdigit() else (0, example_id)
