"""Offline check of label-free rewards on sampled traces, before any training.

Outcome rewards: how well each reward's GRPO advantages align with the true ones, treating each trace's score function
as its own direction. Process rewards at cuts: Brier against the trace's outcome Y, split into the floor E[V(1-V)],
which no reward can remove, and the excess E[(r - V)^2], which bounds the gradient bias.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from data.answers import equal, normalise

RUNS = Path(__file__).resolve().parents[2] / "runs" / "belief_tracking"
PASS_BINS = (0.0, 0.25, 0.5, 0.75, 1.0)


def majority(answers):
    votes = Counter(a for a in answers if a is not None)
    return votes.most_common(1)[0][0] if votes else None


def outcome_rewards(record, rng):
    """True reward and label-free rewards of the question's traces."""
    answers = [normalise(t["answer"]) for t in record["traces"]]
    votes, top = Counter(answers), majority(answers)
    true = np.array([equal(t["answer"], record["answer"]) for t in record["traces"]], dtype=float)
    return true, {"majority vote": np.array([a is not None and a == top for a in answers], dtype=float),
                  "vote share": np.array([(votes[a] - 1) / (len(answers) - 1) if a else 0.0 for a in answers]),
                  "random": rng.integers(0, 2, len(answers)).astype(float)}


def advantages(rewards, normalised):
    centred = rewards - rewards.mean()
    return centred / (rewards.std() + 1e-6) if normalised else centred


def alignment(true, label_free, normalised):
    """Cosine between the summed true and label-free policy gradients."""
    pairs = [(advantages(t, normalised), advantages(r, normalised)) for t, r in zip(true, label_free)]
    dot = sum(a @ b for a, b in pairs)
    return dot / np.sqrt(sum(a @ a for a, _ in pairs) * sum(b @ b for _, b in pairs))


def outcome_report(records, rng):
    rewards = [outcome_rewards(r, rng) for r in records]
    true = [t for t, _ in rewards]
    p = np.array([t.mean() for t in true])
    signal = np.array([0 < x < 1 for x in p])
    print(f"  outcome rewards: {len(records)} questions, mean pass rate {p.mean():.2f}, "
          f"{signal.mean():.0%} with both outcomes (the only ones with a true gradient)")
    edges = " ".join(f"{f'p<{hi:g}':>7}" for hi in PASS_BINS[1:])
    print(f"  {'reward':<15} {'cos':>6} {'cos norm':>8} {'Brier':>6} {'harmful':>8} {'spurious':>9}   "
          f"per-question cos by pass rate {edges}")
    for name in rewards[0][1]:
        hat = [r[name] for _, r in rewards]
        cos = np.array([np.corrcoef(t, h)[0, 1] if s and h.std() else np.nan for t, h, s in zip(true, hat, signal)])
        brier = np.array([np.mean((h - t) ** 2) for t, h in zip(true, hat)])
        push = np.array([np.sum(advantages(h, False) ** 2) for h in hat])
        bins = np.digitize(p, PASS_BINS[1:-1])
        valid = signal & ~np.isnan(cos)
        by_bin = " ".join(f"{np.mean(cos[valid & (bins == b)]):>7.2f}" for b in range(len(PASS_BINS) - 1))
        print(f"  {name:<15} {alignment(true, hat, False):>6.2f} {alignment(true, hat, True):>8.2f} "
              f"{brier.mean():>6.3f} {np.mean(cos[valid] < 0):>8.0%} {push[~signal].sum() / push.sum():>9.0%}   "
              f"{'':>29}{by_bin}")
        print(f"  {'':<15} Spearman(per-question Brier, cos) = {spearmanr(brier[valid], cos[valid])[0]:+.2f}")


def process_report(records):
    """Rewards at cuts against the rollout value V (share of rollouts correct) and the trace's own outcome Y."""
    rows = {"V": [], "Y": [], "cut": [], "forced, own answer": [], "rollouts, majority": [],
            "forced, correct (label)": []}
    for record in records:
        top = majority([normalise(t["answer"]) for t in record["traces"]])
        for t in record["tracked"]:
            own = record["traces"][t["trace"]]["answer"]
            for k, cut in enumerate(sorted(t["cuts"], key=lambda c: c["position"])):
                rows["V"].append(np.mean([equal(a, record["answer"]) for a in cut["rollouts"]]))
                rows["Y"].append(float(equal(own, record["answer"])))
                rows["cut"].append(k)
                rows["forced, own answer"].append(np.mean([equal(a, own) for a in cut["forced"]]))
                # Shares its rollouts with V, so its excess is slightly understated.
                rows["rollouts, majority"].append(np.mean([normalise(a) == top and a is not None
                                                           for a in cut["rollouts"]]))
                rows["forced, correct (label)"].append(np.mean([equal(a, record["answer"]) for a in cut["forced"]]))
    rows = {k: np.array(v, dtype=float) for k, v in rows.items()}
    n = len(records[0]["tracked"][0]["cuts"][0]["rollouts"])
    V, Y, cut = rows.pop("V"), rows.pop("Y"), rows.pop("cut")
    noise = V * (1 - V) / (n - 1)
    half = cut.max() // 2
    print(f"  process rewards at cuts: floor E[V(1-V)] = {np.mean(V * (1 - V) + noise):.3f} "
          f"(early cuts {np.mean((V * (1 - V) + noise)[cut <= half]):.3f}, late {np.mean((V * (1 - V) + noise)[cut > half]):.3f})")
    print(f"  {'reward':<25} {'Brier vs Y':>11} {'excess':>8} {'early':>7} {'late':>7}")
    for name, r in rows.items():
        excess = (r - V) ** 2 - noise
        print(f"  {name:<25} {np.mean((r - Y) ** 2):>11.3f} {excess.mean():>8.3f} "
              f"{excess[cut <= half].mean():>7.3f} {excess[cut > half].mean():>7.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    for path in sorted((RUNS / args.model.replace("/", "__")).glob("*.json")):
        records = json.loads(path.read_text())
        print(f"\n{path.stem}")
        outcome_report(records, rng)
        process_report(records)


if __name__ == "__main__":
    main()
