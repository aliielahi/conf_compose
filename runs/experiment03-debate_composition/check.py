"""Inspect saved debate cells: accuracy before and after, answer resolution, flips, truncation and position bias."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from conf_compose.data import get_task
from conf_compose.debate import DEBATE_STORE, DebateSettings, load_round

SOURCES = ("explicit", "peer_reference", "kept_previous", "inferred", "none")


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


def main():
    args = parse_args()
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
