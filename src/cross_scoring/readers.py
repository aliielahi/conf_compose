"""Who notices an error: readers that solved the problem against readers that did not, and shared against
reader-specific signal.

Localisation of a wrong solution under a score: the share of its earlier (correct) steps that the first wrong step
outscores, 0.5 at chance. Every reader's scores are percentiles over the split, as in evaluate.
"""

import json
import sys
import warnings

import numpy as np

from cross_scoring.evaluate import RUNS, belief_drops, percentile, window_scores
from cross_scoring.solve import SPLITS
from cross_scoring.value import consensus
from data.answers import equal
from data.benchmarks import load as load_benchmark
from data.processbench import load as load_split

BOOTSTRAP = 200


def key(text):
    return " ".join(text.split())


def signals(split, readers, n_items):
    """Per signal and reader: (solutions, steps) error scores as percentiles over the split."""
    out = {"surprise, step mean": {}, "surprise, 32-token": {}, "belief drop": {}}
    for r in readers:
        saved = np.load(RUNS / "scores" / r / f"{split}.npz")
        out["surprise, step mean"][r] = percentile(window_scores(saved, None))
        out["surprise, 32-token"][r] = percentile(window_scores(saved, 32))
        beliefs = np.load(RUNS / "beliefs" / r / f"{split}.npz")
        out["belief drop"][r] = percentile(belief_drops(beliefs, n_items, saved["n_steps"]))
    return out


def localisation(scores, labels):
    """(solutions,): share of earlier steps the first wrong step outscores; nan if not a wrong solution with an earlier
    step or any of those steps is unscored."""
    out = np.full(len(labels), np.nan)
    for i in np.flatnonzero(labels >= 1):
        first, before = scores[i, labels[i]], scores[i, :labels[i]]
        if not np.isnan(first) and not np.isnan(before).any():
            out[i] = np.mean((first > before) + 0.5 * (first == before))
    return out


def demean(values, groups):
    return values - (np.bincount(groups, values) / np.bincount(groups))[groups]


def solver_effect(loc, solved, rng):
    """Effect of 'the reader solved the problem' on localisation, with solution and reader fixed effects (solution
    effects absorbed by demeaning), and its bootstrap standard error over solutions."""
    reader, solution = np.nonzero(~np.isnan(loc))
    y, x = loc[reader, solution], solved[reader, solution].astype(float)
    dummies = np.eye(loc.shape[0])[reader]

    def fit(rows, groups):
        X = np.column_stack([demean(x[rows], groups)] + [demean(d, groups) for d in dummies[rows].T])
        return np.linalg.lstsq(X, demean(y[rows], groups), rcond=None)[0][0]

    ids, groups = np.unique(solution, return_inverse=True)
    members = [np.flatnonzero(groups == g) for g in range(len(ids))]
    draws = []
    for _ in range(BOOTSTRAP):
        picks = rng.integers(len(ids), size=len(ids))
        rows = np.concatenate([members[p] for p in picks])
        draws.append(fit(rows, np.repeat(np.arange(len(picks)), [len(members[p]) for p in picks])))
    return fit(np.arange(len(y)), groups), np.std(draws)


def panel(by_reader, readers, include):
    """Median over the readers included for each solution (include: readers x solutions)."""
    stacked = np.stack([by_reader[r] for r in readers])
    return np.nanmedian(np.where(include[:, :, None], stacked, np.nan), axis=0)


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)  # medians and means over empty reader sets
    rng = np.random.default_rng(0)
    readers = sorted(p.name for p in (RUNS / "beliefs").iterdir())
    for split in sys.argv[1:] or SPLITS:
        rows = load_split(split)
        labels = np.array([r["label"] for r in rows])
        gold = {key(r["problem"]): r["answer"] for r in load_benchmark(split)}
        target = consensus(split)
        answers = {r: json.loads((RUNS / "answers" / r / f"{split}.json").read_text()) for r in readers}
        has_gold = np.array([key(r["problem"]) in gold for r in rows])
        solved = np.array([[equal(answers[m].get(r["problem"]), gold.get(key(r["problem"]))) for r in rows]
                           for m in readers])
        agrees = np.array([[equal(answers[m].get(r["problem"]), target.get(r["problem"])) for r in rows]
                           for m in readers])
        by_signal = signals(split, readers, len(rows))
        print(f"\n{split}: {np.sum(labels >= 1)} wrong solutions with an earlier step, gold answer for "
              f"{has_gold.mean():.0%}, readers solved {solved[:, has_gold].mean():.0%} of (reader, problem) pairs")

        print("  localisation of the first wrong step (0.5 = chance)")
        print(f"  {'signal':<22} {'solved':>7} {'unsolved':>9} {'within reader':>14} {'fixed effects':>17}")
        for name, by_reader in by_signal.items():
            loc = np.stack([localisation(by_reader[r], labels) for r in readers])
            loc[:, ~has_gold] = np.nan
            gaps = [np.nanmean(l[s]) - np.nanmean(l[~s]) for l, s in zip(loc, solved)]
            effect, se = solver_effect(loc, solved, rng)
            print(f"  {name:<22} {np.nanmean(loc[solved]):>7.3f} {np.nanmean(loc[~solved]):>9.3f} "
                  f"{np.nanmean(gaps):>+14.3f} {effect:>+10.3f} ± {se:.3f}")

        print("  shared against reader-specific signal (specific = reader minus the mean of the other readers)")
        print(f"  {'signal':<22} {'shared variance':>16} {'shared':>7} {'specific, solved':>17} {'unsolved':>9}")
        for name, by_reader in by_signal.items():
            stacked = np.stack([by_reader[r] for r in readers])
            shared = np.nanmean(stacked, axis=0)
            valid = ~np.isnan(stacked)
            share = np.var(np.broadcast_to(shared, stacked.shape)[valid]) / np.var(stacked[valid])
            others = (np.nansum(stacked, axis=0) - np.nan_to_num(stacked)) / np.maximum(valid.sum(0) - 1, 1)
            specific = np.stack([localisation(stacked[m] - others[m], labels) for m in range(len(readers))])
            specific[:, ~has_gold] = np.nan
            shared_loc = localisation(shared, labels)[has_gold]
            print(f"  {name:<22} {share:>16.2f} {np.nanmean(shared_loc):>7.3f} "
                  f"{np.nanmean(specific[solved]):>17.3f} {np.nanmean(specific[~solved]):>9.3f}")

        print("  panels (median over readers)")
        print(f"  {'signal':<22} {'all':>7} {'solvers (gold)':>15} {'agree with consensus':>21}")
        everyone = np.ones_like(solved)
        for name, by_reader in by_signal.items():
            cells = [np.nanmean(localisation(panel(by_reader, readers, include), labels)[has_gold])
                     for include in (everyone, solved, agrees)]
            print(f"  {name:<22} {cells[0]:>7.3f} {cells[1]:>15.3f} {cells[2]:>21.3f}")

        print("  per reader, surprise step mean")
        print(f"  {'reader':<45} {'solve rate':>10} {'solved':>7} {'unsolved':>9}")
        for m, r in enumerate(readers):
            loc = localisation(by_signal["surprise, step mean"][r], labels)
            loc[~has_gold] = np.nan
            print(f"  {r:<45} {solved[m, has_gold].mean():>10.0%} {np.nanmean(loc[solved[m]]):>7.3f} "
                  f"{np.nanmean(loc[~solved[m]]):>9.3f}")


if __name__ == "__main__":
    main()
