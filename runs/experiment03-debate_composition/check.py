"""Inspect saved debate cells: completeness and signal coverage, then accuracy, answer resolution and position bias."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from conf_compose.data import get_task
from conf_compose.debate import DEBATE_STORE, DebateSettings, load_round, shared_ids
from conf_compose.debate.store import read_rows

SOURCES = ("explicit", "peer_reference", "kept_previous", "inferred", "none")
SAMPLES = "consistency_t0.7"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--out-dir", type=Path, default=DEBATE_STORE)
    parser.add_argument("--match", default="", help="substring of a cell name, e.g. _n20")
    return parser.parse_args()


def saved_cells(args):
    for path in sorted(args.out_dir.glob("*/*/settings.json")):
        settings = json.loads(path.read_text())["settings"]
        if (args.tasks and settings["task"] not in args.tasks) or args.match not in path.parent.name:
            continue
        fields = {key: value for key, value in settings.items() if key not in ("digest", "name")}
        yield DebateSettings(**{**fields, "group": tuple(fields["group"])}), path.parent


def rounds_written(directory):
    return sorted(int(path.name.split("_")[1]) for path in directory.glob("round_*") if path.is_dir())


def majority(task, answers):
    counts = Counter()
    for answer in answers:
        if answer is not None:
            key = next((k for k in counts if task.equivalent(k, answer)), answer)
            counts[key] += 1
    return counts.most_common(1)[0][0] if counts else None


def completeness(args):
    """Every file present, every row there, and every signal and candidate score filled in."""
    print(f"{'cell':<58}{'rows':>6}{'files':>7}{'cons':>7}{'seq':>7}{'debias':>7}{'5 smp':>7}"
          f"{'scored':>8}{'direct':>8}{'pool':>6}  status")
    problems = 0
    for settings, directory in saved_cells(args):
        rounds = [r for r in rounds_written(directory) if r > 0]
        last = max(rounds, default=0)
        expected = len(shared_ids({m: load_round(settings, m, 0) for m in settings.group}, settings.limit))
        paths = [settings.round_path(m, r, args.out_dir) for m in settings.group for r in rounds] + \
                [settings.scores_path(m, last, args.out_dir) for m in settings.group]
        present = [path for path in paths if path.exists()]
        turns = [row for path in present if path.parent.name.startswith("round_") for row in read_rows(path)]
        scores = [row for path in present if path.parent.name.startswith("candidates") for row in read_rows(path)]
        answered = [t for t in turns if t["prediction"] is not None]
        live = [t for t in turns if not t.get("error")]

        def share(rows, test):
            return sum(map(test, rows)) / len(rows) if rows else 0.0

        cells = {
            "cons": share(answered, lambda t: t["confidence"].get(SAMPLES) is not None),
            "seq": share(live, lambda t: t["confidence"].get("seq_response") is not None),
            "debias": share(live, lambda t: t["confidence"].get("seq_response_debiased") is not None),
            "samples": share(live, lambda t: len((t.get("sampled_answers") or {}).get(SAMPLES, [])) == 5),
            "direct": share(scores, lambda s: all(c.get("direct") for c in s["candidate_scores"]["candidates"])),
        }
        wanted_scores = len(settings.group) * (last + 1) * expected - sum(1 for t in turns if t.get("error"))
        rows_ok = len(turns) == len(settings.group) * len(rounds) * expected and len(scores) == wanted_scores
        healthy = len(present) == len(paths) and rows_ok and min(cells.values()) >= 0.99
        problems += not healthy
        pool = sum(len(s["candidate_scores"]["candidates"]) for s in scores) / max(len(scores), 1)
        print(f"{directory.name[:57]:<58}{expected:>6}{len(present):>4}/{len(paths):<2}{cells['cons']:>7.3f}"
              f"{cells['seq']:>7.3f}{cells['debias']:>7.3f}{cells['samples']:>7.3f}{len(scores):>8}"
              f"{cells['direct']:>8.3f}{pool:>6.1f}  {'ok' if healthy else 'CHECK'}")
    print(f"\n{problems} cell(s) need a look" if problems else "\nall cells complete")


def main():
    args = parse_args()
    completeness(args)
    print()
    totals = defaultdict(Counter)
    print(f"{'cell':<58}{'r':>2}{'n':>6}{'acc0':>7}{'acc':>7}{'vote0':>7}{'vote':>7}{'chg':>6}"
          f"{'w>r':>5}{'r>w':>5}{'trunc':>6}{'ovf':>5}  answer source")
    for settings, directory in saved_cells(args):
        task = get_task(settings.task)
        base = {model: load_round(settings, model, 0) for model in settings.group}
        for round_index in rounds_written(directory):
            if not all(settings.round_path(m, round_index, args.out_dir).exists() for m in settings.group):
                continue
            current = {m: load_round(settings, m, round_index, out=args.out_dir) for m in settings.group}
            ids = list(next(iter(current.values())))
            turns = [current[m][i] for m in settings.group for i in ids]
            before = [base[m][i] for m in settings.group for i in ids]
            gold = {i: base[settings.group[0]][i]["gold"] for i in ids}
            vote0 = sum(task.equivalent(majority(task, [base[m][i]["prediction"] for m in settings.group]) or "", gold[i])
                        for i in ids) / len(ids)
            vote = sum(task.equivalent(majority(task, [current[m][i]["prediction"] for m in settings.group]) or "", gold[i])
                       for i in ids) / len(ids)
            sources = Counter(t["answer_source"] for t in turns)
            fixed = sum(not b["correct"] and t["correct"] for b, t in zip(before, turns))
            broke = sum(b["correct"] and not t["correct"] for b, t in zip(before, turns))
            row = Counter(n=len(turns), correct=sum(t["correct"] for t in turns),
                          correct0=sum(b["correct"] for b in before), changed=sum(t["changed"] for t in turns),
                          fixed=fixed, broke=broke, trunc=sum(t["finish_reason"] == "length" for t in turns),
                          overflow=sum(t.get("error") == "context_overflow" for t in turns), **sources)
            totals[(settings.task, round_index)] += row
            print(f"{directory.name[:57]:<58}{round_index:>2}{len(turns):>6}{row['correct0'] / len(turns):>7.3f}"
                  f"{row['correct'] / len(turns):>7.3f}{vote0:>7.3f}{vote:>7.3f}{row['changed'] / len(turns):>6.2f}"
                  f"{fixed:>5}{broke:>5}{row['trunc']:>6}{row['overflow']:>5}  "
                  + " ".join(f"{s}={sources[s]}" for s in SOURCES if sources[s]))
            position_bias(settings, current, totals[(settings.task, round_index)])
    report(totals)


def position_bias(settings, current, total):
    """When an agent switches to an answer exactly one peer held, how often was that peer shown first?"""
    for model, rows in current.items():
        for row in rows.values():
            if row["changed"] and len(row["adopted_from"]) == 1 and len(row["peer_order"]) > 1:
                total["switches"] += 1
                total["first_shown"] += row["peer_order"][0] == row["adopted_from"][0]
                total["first_expected"] += 1 / len(row["peer_order"])


def report(totals):
    print(f"\n{'task':<12}{'r':>2}{'turns':>8}{'acc0':>7}{'acc':>7}{'changed':>9}{'fixed':>7}{'broke':>7}"
          f"{'trunc':>7}{'overflow':>9}  first-shown peer adopted / expected")
    for (task, round_index), row in sorted(totals.items()):
        n = row["n"]
        bias = (f"{row['first_shown'] / row['switches']:.2f} / {row['first_expected'] / row['switches']:.2f}"
                f" over {row['switches']} switches" if row["switches"] else "-")
        print(f"{task:<12}{round_index:>2}{n:>8}{row['correct0'] / n:>7.3f}{row['correct'] / n:>7.3f}"
              f"{row['changed'] / n:>9.3f}{row['fixed']:>7}{row['broke']:>7}{row['trunc']:>7}{row['overflow']:>9}  {bias}")
        unresolved = row["none"] + row["inferred"]
        if unresolved:
            print(f"{'':<14}answers not stated explicitly: {row['inferred']} inferred, {row['none']} unresolved, "
                  f"{row['peer_reference']} via a peer, {row['kept_previous']} kept")


if __name__ == "__main__":
    main()
