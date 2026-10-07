"""Load exactly the panels, labels and outer splits used by the paper tables.

Only standard-library imports: auditing does not require PyTorch or a GPU.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROJECT = ROOT.parents[2]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(2**20), b""):
            h.update(part)
    return h.hexdigest()


def id_digest(ids):
    return hashlib.sha256(json.dumps(sorted(ids), separators=(",", ":")).encode()).hexdigest()


def fitting_id(task, qid, fraction, seed):
    key = json.dumps([task, seed, qid]).encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / 2**64 < fraction


def internal_splits(task, fit_ids, seed, fraction=0.2):
    """Split by question, independently of panel, estimator and correctness."""
    ids = sorted(fit_ids, key=lambda q: digest(["cage-validation", task, seed, q]))
    if len(ids) < 5:
        raise ValueError(f"{task}: need at least five outer-fitting questions")
    n = max(1, min(len(ids) - 2, round(len(ids) * fraction)))
    return set(ids[n:]), set(ids[:n])


def safe_output(path):
    path = Path(path).resolve()
    if path == ROOT or ROOT not in path.parents:
        raise ValueError(f"Output must be a subdirectory of {ROOT}: {path}")
    return path


def project_api(project):
    # Never leave __pycache__ files in the parent project.
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(Path(project) / "src"))
    from conf_compose.data import get_task
    return get_task


def majority(task, answers):
    """Task equivalence; vote ties go to the first model in the ordered panel."""
    reps, counts, normalized = [], [], []
    for answer in answers:
        if answer is None or str(answer).strip() == "":
            normalized.append(None)
            continue
        answer = str(answer)
        index = next((i for i, rep in enumerate(reps) if task.equivalent(answer, rep)), None)
        if index is None:
            index = len(reps)
            reps.append(answer)
            counts.append(0)
        counts[index] += 1
        normalized.append(reps[index])
    target = reps[max(range(len(reps)), key=lambda i: counts[i])] if reps else None
    return target, normalized


def read_records(path):
    """Discard large sampled responses/candidate-score arrays immediately."""
    out = {}
    with Path(path).open() as f:
        for line in f:
            raw = json.loads(line)
            qid = raw["id"]
            if qid in out:
                raise ValueError(f"Duplicate question {qid} in {path}")
            row = {k: raw.get(k) for k in
                   ("id", "question", "gold", "prediction", "correct", "options", "choices")}
            lp = (raw.get("token_logprobs") or {}).get("response")
            row["mean_logprob"] = (sum(lp) / len(lp) if lp and
                all(isinstance(v, (float, int)) and math.isfinite(v) for v in lp) else None)
            out[qid] = row
    return out


def find_source(project, task, model, row):
    token = f"s70v{row['voter']}_k{row['n_samples']}_"
    candidates = [p for p in (project / "results/inferences" / task).glob(
        model.replace("/", "__") + "--*/test.jsonl")
        if token in p.parent.name and row["match"] in p.parent.name]
    if len(candidates) != 1:
        raise ValueError(f"{task}/{model}: expected one saved cell, got {candidates}")
    return candidates[0]


def load_bundle(project=DEFAULT_PROJECT, tasks=None, panel=None, val_fraction=0.2, seed=0):
    project = Path(project).resolve()
    get_task = project_api(project)
    table_dir = project / "paper_results/results/voting_protocol"
    audit_path, atomic_path = table_dir / "audit.json", table_dir / "atomic.csv"
    cells = json.loads(audit_path.read_text())
    cells = [c for c in cells if (not tasks or c["task"] in tasks)
             and (not panel or c["models"] == "|".join(panel))]
    if not cells:
        raise ValueError("No matching paper-table cells; specify an existing ordered panel")
    if tasks and set(tasks) != {c["task"] for c in cells}:
        raise ValueError("Some requested tasks have no matching paper-table cells")
    specs = {}
    provenance = {str(p.relative_to(project)): file_hash(p) for p in (audit_path, atomic_path)}
    for estimator in sorted({c["estimator"] for c in cells}):
        path = project / f"results/voting_atomic/learned_panels_v1/{estimator}.csv"
        provenance[str(path.relative_to(project))] = file_hash(path)
        with path.open() as handle:
            for r in csv.DictReader(handle):
                specs[(r["task"], r["models"], estimator)] = r
    rows, sources, splits, group_summaries = [], {}, {}, []
    selected = sorted({(c["task"], c["models"]) for c in cells})
    for task_name, models in selected:
        task = get_task(task_name)
        group_cells = [c for c in cells if (c["task"], c["models"]) == (task_name, models)]
        spec = specs[(task_name, models, group_cells[0]["estimator"])]
        if spec["fit_split"] != "holdout" or int(spec["voter"]) != 0:
            raise ValueError("This adapter supports the existing voter-0 holdout experiment")
        model_ids = json.loads(spec["model_ids"])
        if "|".join(m.split("/")[-1] for m in model_ids) != models:
            raise ValueError("Model order/alias mismatch")
        records = []
        for model in model_ids:
            key = (task_name, model)
            path = find_source(project, task_name, model, spec)
            if key not in sources:
                print(f"Reading {task_name}/{model}", flush=True)
                sources[key] = (path, file_hash(path), read_records(path))
            saved_path, actual_hash, recs = sources[key]
            if saved_path != path:
                raise ValueError(f"Mixed source cells for {key}")
            for c in group_cells:
                other = specs[(task_name, models, c["estimator"])]
                if json.loads(other["input_hashes"])[model]["test"] != actual_hash:
                    raise ValueError(f"Source hash changed: {path}; regenerate paper inputs first")
            provenance[str(path.relative_to(project))] = actual_hash
            records.append(recs)
        ids = set.intersection(*(set(r) for r in records))
        fit = {q for q in ids if fitting_id(task_name, q, float(spec["fit_fraction"]), int(spec["fit_seed"]))}
        evaluation = ids - fit
        for c in group_cells:
            for label, values in (("fit", fit), ("eval", evaluation), ("matched", c["matched_ids"])):
                if id_digest(values) != c[label + "_ids_hash"]:
                    raise ValueError(f"{task_name}/{models}: {label} IDs do not match paper audit")
            if not set(c["matched_ids"]) <= evaluation:
                raise ValueError("Paper evaluation mask contains fitting questions")
        train, validation = internal_splits(task_name, fit, seed, val_fraction)
        this_split = dict(train=sorted(train), validation=sorted(validation), evaluation=sorted(evaluation))
        if task_name in splits and splits[task_name] != this_split:
            raise ValueError(f"{task_name}: panel-dependent question splits would leak labels")
        splits[task_name] = this_split
        missing, split_counts, group_rows = [], Counter(), []
        for qid in sorted(ids):
            members = [r[qid] for r in records]
            first = members[0]
            for m in members[1:]:
                if any(m[k] != first[k] for k in ("question", "gold", "options", "choices")):
                    raise ValueError(f"{task_name}/{qid}: question, gold or option mapping differs between models")
            target, answers = majority(task, [r["prediction"] for r in members])
            for m in members:
                expected = m["prediction"] is not None and task.equivalent(str(m["prediction"]), str(m["gold"]))
                if bool(m["correct"]) != expected:
                    raise ValueError(f"{task_name}/{qid}: saved correctness disagrees with current task grader")
            if any(a is None or m["mean_logprob"] is None for a, m in zip(answers, members)):
                missing.append(qid)
                continue
            split = "train" if qid in train else "validation" if qid in validation else "evaluation"
            # Labels are kept separately; features.py is only given training labels for W.
            row = dict(task=task_name, models=models, model_ids=model_ids, id=qid,
                       question=first["question"], target=target, answers=answers,
                       mean_logprobs=[m["mean_logprob"] for m in members], split=split,
                       member_correct=[int(m["correct"]) for m in members],
                       correct=int(task.equivalent(target, str(first["gold"]))))
            rows.append(row)
            group_rows.append(row)
            split_counts[split] += 1
        for c in group_cells:
            lost = set(c["matched_ids"]) & set(missing)
            if lost:
                raise ValueError(f"{task_name}/{models}: {len(lost)} paper-matched examples lack CAGE features; "
                                 "cannot produce a row on a different evaluation mask")
            matched = set(c["matched_ids"])
            scored = [r for r in group_rows if r["id"] in matched]
            accuracy = sum(r["correct"] for r in scored) / len(scored)
            if abs(accuracy - c["vote_accuracy"]) > 1e-12:
                raise ValueError(f"{task_name}/{models}: reconstructed majority accuracy differs from paper")
        group_summaries.append(dict(task=task_name, models=models, counts=dict(split_counts),
                                    missing_feature_ids=missing))
    return dict(project=project, rows=rows, cells=cells, splits=splits,
                sources=provenance, groups=group_summaries, atomic_path=atomic_path)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metric_module(project):
    return load_module(Path(project) / "src/conf_compose/utils/metrics.py", "voting_original_metrics")
