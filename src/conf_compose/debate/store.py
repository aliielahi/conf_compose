"""Reading round 0 from the inference store and writing debate rounds in the same record schema."""

from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from conf_compose.pipelines.inference import STORE

from .settings import DEBATE_STORE, DebateSettings

Record = Dict[str, Any]


def round0_cell(task: str, model: str, voter: int = 0, match: str = "_cs7s", temperature: float = 0.7,
                store: Path = STORE) -> Path:
    """The one store cell voting used for this model, so debate starts from exactly the same answers."""
    decoding = f"s{int(round(temperature * 100)):02d}v{voter}_"
    cells = [path.parent for path in sorted(Path(store, task).glob(f"vllm__{model}--{decoding}*/test.jsonl"))
             if match in path.parent.name]
    if len(cells) != 1:
        listing = "\n".join(f"  {cell.name}" for cell in cells) or "  none"
        raise FileNotFoundError(f"need exactly one round-0 cell for {task}/{model} ({decoding}, {match}):\n{listing}")
    return cells[0]


def round0_budget(task: str, models: Sequence[str], voter: int = 0, match: str = "_cs7s",
                  store: Path = STORE) -> int:
    """The largest token budget round 0 allowed, retries included, so debate truncates no more than it did."""
    budgets = []
    for model in models:
        saved = json.loads((round0_cell(task, model, voter, match, store=store) / "settings.json").read_text())
        budgets.append(saved["settings"]["retry_max_tokens"] or saved["settings"]["max_tokens"])
    return max(budgets)


def round0_confidence_config(task: str, model: str, voter: int = 0, match: str = "_cs7s",
                             store: Path = STORE) -> Dict[str, Any]:
    return json.loads((round0_cell(task, model, voter, match, store=store) / "settings.json").read_text())["config"]


def read_rows(path: Path) -> List[Record]:
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_rows(path: Path, rows: Sequence[Record]) -> None:
    """Written to a temporary name and renamed, so a file that exists is always complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(json.dumps(row) + "\n" for row in rows))
    os.replace(temporary, path)


def file_hash(path: Path) -> str:
    path = Path(path)
    return _hash(str(path), path.stat().st_mtime_ns)


@lru_cache(maxsize=None)
def _hash(path: str, modified: int) -> str:
    value = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def round_path(settings: DebateSettings, model: str, round_index: int, store: Path = STORE,
               out: Path = DEBATE_STORE) -> Path:
    if round_index == 0:
        return round0_cell(settings.task, model, settings.voter, settings.match, settings.temperature, store) / "test.jsonl"
    return settings.round_path(model, round_index, out)


def load_round(settings: DebateSettings, model: str, round_index: int, store: Path = STORE,
               out: Path = DEBATE_STORE) -> Dict[str, Record]:
    """One agent's records for one round, keyed by question ID, in file order."""
    rows = read_rows(round_path(settings, model, round_index, store, out))
    return {row["id"]: row for row in rows}


def shared_ids(rounds: Dict[str, Dict[str, Record]], limit: Optional[int] = None) -> List[str]:
    """Questions every agent answered, in the first agent's file order; a limit keeps the first ones."""
    first, *rest = rounds.values()
    ids = [example_id for example_id in first if all(example_id in other for other in rest)]
    return ids[:limit] if limit else ids


def load_with_scores(settings: DebateSettings, model: str, round_index: int, through: int,
                     store: Path = STORE, out: Path = DEBATE_STORE) -> Dict[str, Record]:
    """A round's records with the debate's own candidate scores attached, in the inference-store schema."""
    records = load_round(settings, model, round_index, store, out)
    for row in read_rows(settings.scores_path(model, through, out)):
        if row["round"] == round_index and row["id"] in records:
            records[row["id"]] = {**records[row["id"]], "candidate_scores": row["candidate_scores"]}
    return records


def write_settings(settings: DebateSettings, store: Path = STORE, out: Path = DEBATE_STORE) -> Path:
    """The cell's settings and the exact round-0 files it builds on, hashed so a changed source is caught."""
    sources = {}
    for model in settings.group:
        path = round_path(settings, model, 0, store, out)
        sources[model] = {"path": str(path), "sha256": file_hash(path)}
    target = settings.directory(out) / "settings.json"
    payload = {"settings": settings.to_dict(), "round0": sources}
    if target.exists():
        saved = json.loads(target.read_text())
        changed = [m for m in settings.group if saved["round0"][m]["sha256"] != sources[m]["sha256"]]
        if changed:
            raise ValueError(f"round-0 source changed since {target} was written: {changed}")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n")
    return target
