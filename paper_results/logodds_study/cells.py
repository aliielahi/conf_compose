"""Per-group confidence matrices for both estimators with question IDs and each stream's log-probabilities over the
whole answer space, saved as cells.pkl."""

import importlib.util
import json
import math
import pickle
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from conf_compose.composition.candidates import OTHER, candidate_set, label_space, support_of
from conf_compose.composition.debate_records import load_cell
from conf_compose.composition.evidence import Stream
from conf_compose.data import Example, get_task

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


voting = load("voting_data", ROOT / "paper_results/codes/voting_protocol/data.py")
replay = load("seq_vs_cons_extract", ROOT / "paper_results/seq_vs_cons/extract.py")
STORE = ROOT / "results/inferences"
SAMPLES = 5
# Confidence estimator -> (pipeline estimator, scoring context, sequence score), as in per_system_results.
ESTIMATORS = {"cons": ("cons", "direct", "norm_sum"), "seq_reasoned": ("seq", "reasoned", "norm_sum")}


def states_of(task, record, answers):
    """The answer space: the question's options, else the task's labels, plus every other answer seen (rare invalid labels)."""
    found = list(record.get("options") or label_space(task) or [])
    for answer in answers:
        if answer is not None and not any(task.equivalent(answer, seen) for seen in found):
            found.append(answer)
    return found


def sequence_distribution(task, record, states, context):
    """Log-softmax over the record's scored candidates, placed on the shared states; unscored states get -inf."""
    candidates = (record.get("candidate_scores") or {}).get("candidates") or []
    if not candidates or record.get("error"):
        return None
    logs = [sum((c.get(context) or {}).get("logprobs") or [math.nan]) for c in candidates]
    if not all(map(math.isfinite, logs)):
        return None
    logs = np.asarray(logs) - logsumexp(logs)
    out = np.full(len(states), -np.inf)
    for candidate, value in zip(candidates, logs):
        index = next((i for i, s in enumerate(states) if task.equivalent(s, candidate["answer"])), None)
        if index is not None:
            out[index] = np.logaddexp(out[index], value)
    return out


def consistency_distribution(task, samples, states):
    """Log of uniform-Dirichlet smoothed sample frequencies over the states (open tasks add an OTHER state)."""
    support = support_of(task, samples[:SAMPLES], [s for s in states if s != OTHER])
    if support.valid == 0:
        return None
    return np.log([(support.counts.get(s, 0) + 1 / len(states)) / (support.valid + 1) for s in states])


def question(task, estimator, context, streams, target):
    """(streams x states) log-probabilities with the target state first, or None if any stream has none."""
    records = [record for record, _ in streams]
    answers = [stream.answer for _, stream in streams]
    if estimator == "seq":
        answers += [c["answer"] for r in records for c in (r.get("candidate_scores") or {}).get("candidates") or []]
    states = states_of(task, records[0], [target, *answers])
    if estimator == "cons" and not (records[0].get("options") or label_space(task)):
        states = [*states, OTHER]
    rows = [sequence_distribution(task, r, states, context) if estimator == "seq"
            else consistency_distribution(task, s.samples, states) for r, s in streams]
    if any(row is None for row in rows):
        return None
    first = next(i for i, s in enumerate(states) if task.equivalent(s, target))
    order = [first, *[i for i in range(len(states)) if i != first]]
    return np.asarray(rows)[:, order]


def stream(record, model, agent, round_index):
    return Stream(f"{record['id']}:r{round_index}:a{agent}", agent, round_index, model, record["prediction"],
                  (record.get("sampled_answers") or {}).get("consistency_t0.7", []))


def pack(estimator, setting, task, models, rows):
    """`rows` maps split -> list of (id, streams' confidences, label, log-probabilities)."""
    cell = {"estimator": estimator, "setting": setting, "task": task, "models": models,
            "n_models": len(models.split("|"))}
    for split, kept in rows.items():
        cell[split] = np.asarray([r[1] for r in kept], dtype=float)
        cell[f"{split}_labels"] = np.asarray([r[2] for r in kept], dtype=float)
        cell[f"{split}_ids"] = [r[0] for r in kept]
        cell[f"{split}_logp"] = [r[3] for r in kept]
    return cell


def voting_cells(name):
    """Fitting questions need valid shared confidences; evaluation ones are the pipeline's matched set, as in replay.voting_rows."""
    estimator, context, seq_score = ESTIMATORS[name]
    matched = {(c["task"], c["models"]): set(c["matched_ids"])
               for c in json.loads(replay.VOTING_AUDIT.read_text()) if c["estimator"] == estimator}
    atomic = voting.atomic_module()
    sources = voting.SourceStore(STORE)
    for row in voting.load_csvs(replay.POOL_DIR, [estimator]):
        group, task = json.loads(row["model_ids"]), get_task(row["task"])
        args = voting.row_arguments(row, STORE)
        args.context, args.seq_score = context, seq_score
        records = sources.get(atomic, row, group, "test")
        items = atomic.group_items(records, group)
        fitting, evaluation = atomic.split_items(row["task"], items, float(row["fit_fraction"]), int(row["fit_seed"]))
        rows = {"fit": [], "eval": []}
        for split, subset in (("fit", fitting), ("eval", evaluation)):
            for item in subset:
                if split == "eval" and item.example_id not in matched[row["task"], row["models"]]:
                    continue
                target, own, scores = atomic.confidence_scores(task, item, args, records)
                needed = scores if split == "fit" else [*scores, *own]
                if target is None or not all(map(voting.valid_probability, needed)):
                    continue
                streams = [(records[s.model][item.example_id], s) for s in item.round_streams((0,))]
                label = float(task.is_correct(target, Example(item.example_id, item.question, item.gold)))
                rows[split].append((item.example_id, scores, label, question(task, estimator, context, streams, target)))
        yield pack(name, "voting", row["task"], row["models"], rows)


def debate_jobs(name):
    estimator, context, seq_score = ESTIMATORS[name]
    runs = replay.CONSISTENCY_RUNS if estimator == "cons" else dict.fromkeys((1, 2), replay.sequence_runs()[context, seq_score])
    jobs = []
    for round_index, run in runs.items():
        sources = json.loads((replay.DEBATE_DIR / run / "manifest.json").read_text())["identity"]["sources"]
        directories = {Path(p).parent.name: Path(p).parent for p in sources if p.endswith("settings.json")}
        for path in sorted((replay.DEBATE_DIR / run).glob(f"*/*/round_{round_index}/{estimator}/predictions.jsonl")):
            directory = directories[path.parents[2].name]
            jobs.append((name, round_index, path, directory if directory.is_absolute() else ROOT / directory))
    return jobs


def debate_cell(job):
    """Post-debate then round-0 streams in the final answer, on the questions replay.debate_rows keeps."""
    warnings.filterwarnings("ignore")
    name, round_index, path, directory = job
    estimator, context, _ = ESTIMATORS[name]
    audit = json.loads(path.with_name("audit.json").read_text())
    predictions = {r["id"]: r for r in map(json.loads, path.open())}
    debate = load_cell(directory, round_index, STORE, estimator == "seq")
    task, models = get_task(debate.task), debate.models

    def values(q):
        found = predictions[q]["post_debate"]["shared"] + predictions[q]["initial_on_final"]
        return found if all(v is not None for v in found) else None

    rows = {}
    for split, ids in (("fit", audit["fit_scored_ids"]), ("eval", audit["matched_ids"])):
        rows[split] = []
        for q in ids:
            if values(q) is None:
                continue
            streams = ([(debate.current[m][q], stream(debate.current[m][q], m, a, round_index)) for a, m in enumerate(models)]
                       + [(debate.initial[m][q], stream(debate.initial[m][q], m, a, 0)) for a, m in enumerate(models)])
            target = predictions[q]["post_debate"]["target"]
            rows[split].append((q, values(q), float(predictions[q]["correct"]),
                                question(task, estimator, context, streams, target)))
    return pack(name, f"debate_r{round_index}", audit["task"], audit["models"], rows)


def main():
    jobs = [job for name in ESTIMATORS for job in debate_jobs(name)]
    with Pool() as pool:
        debates = pool.map_async(debate_cell, jobs)
        cells = [cell for name in ESTIMATORS for cell in voting_cells(name)]
        cells += debates.get()
    with (HERE / "cells.pkl").open("wb") as handle:
        pickle.dump(cells, handle)
    missing = sum(d is None for c in cells for split in ("fit", "eval") for d in c[f"{split}_logp"])
    print(f"wrote {len(cells)} cells; questions without a full distribution: {missing}")


if __name__ == "__main__":
    main()
