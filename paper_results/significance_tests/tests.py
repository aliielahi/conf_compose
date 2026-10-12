"""Paired significance tests over the 75 (dataset x group) blocks per protocol: exact sign tests with Holm correction
and Friedman ranks with the Nemenyi critical difference. Writes significance.csv, rankings.csv and LaTeX tables."""

import numpy as np
import pandas as pd
from scipy import stats

from blocks import HERE, PROTOCOLS, build

PROPOSED = "Log-odds Learned Weights"
BASELINES = ("Best Agent", "CAGE-CAL", "LLM Elicited")
# Lower p-Brier and higher AUARC are better; deltas are oriented so that positive favors the first system.
METRICS = {"auarc": ("AUARC", 1), "p_brier": ("p-Brier", -1)}
# Studentized range over sqrt(2) at alpha = 0.05, by number of systems (Demsar, 2006).
NEMENYI = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031}
KEY = ["task", "group"]


def deltas(frame, first, second, metric):
    """Per block, first minus second in points, signed so positive favors first."""
    wide = frame.pivot_table(index=KEY, columns="system", values=metric)
    return 100 * METRICS[metric][1] * (wide[first] - wide[second]).dropna()


def sign_test(values, sided="greater"):
    wins, losses = int((values > 0).sum()), int((values < 0).sum())
    upper = stats.binom.sf(wins - 1, wins + losses, 0.5)
    p = upper if sided == "greater" else min(1.0, 2 * min(upper, stats.binom.cdf(wins, wins + losses, 0.5)))
    return {"median": float(values.median()), "wins": wins, "losses": losses, "ties": int((values == 0).sum()), "p": p}


def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adjusted = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    out = np.empty_like(p)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def comparisons(frame):
    """Every planned comparison as one row per metric; Holm runs within each family and metric."""
    rows = []
    consistency = frame[frame.estimator == "consistency"]
    for protocol in PROTOCOLS:
        here = consistency[consistency.protocol == protocol]
        for baseline in BASELINES:
            if baseline in set(here.system):
                rows += [{"family": "baselines", "estimator": "consistency", "protocol": protocol, "first": PROPOSED,
                          "second": baseline, "metric": m, **sign_test(deltas(here, PROPOSED, baseline, m))}
                         for m in METRICS]
    # Bayes against probability pooling, both learned; two-sided because either direction is informative.
    for estimator in ("consistency", "sequence"):
        for protocol in PROTOCOLS:
            here = frame[(frame.estimator == estimator) & (frame.protocol == protocol)]
            rows += [{"family": f"spaces:{estimator}", "estimator": estimator, "protocol": protocol, "first": PROPOSED,
                      "second": "Prob. Learned Weights", "metric": m,
                      **sign_test(deltas(here, PROPOSED, "Prob. Learned Weights", m), "two-sided")} for m in METRICS]
    # Does interaction enlarge the gain of composition over the group's best agent?
    for later, earlier in (("debate_r1", "voting"), ("debate_r2", "debate_r1")):
        for m in METRICS:
            gain = {p: deltas(consistency[consistency.protocol == p], PROPOSED, "Best Agent", m) for p in (later, earlier)}
            rows.append({"family": "interaction", "estimator": "consistency", "protocol": later, "first": later,
                         "second": earlier, "metric": m, **sign_test((gain[later] - gain[earlier]).dropna())})
    result = pd.DataFrame(rows)
    result["p_holm"] = result.groupby(["family", "metric"]).p.transform(holm)
    return result


def rankings(frame):
    """Average rank of every consistency system per protocol and metric (1 = best), with Friedman's test."""
    rows = []
    for protocol in PROTOCOLS:
        here = frame[(frame.estimator == "consistency") & (frame.protocol == protocol)]
        for metric, (name, sign) in METRICS.items():
            wide = here.pivot_table(index=KEY, columns="system", values=metric).dropna(axis=1)
            ranks = pd.DataFrame(stats.rankdata(-sign * wide.to_numpy(), axis=1), columns=wide.columns)
            k, n = wide.shape[1], wide.shape[0]
            rows.append({"protocol": protocol, "metric": name, "blocks": n, "systems": k,
                         "friedman_p": stats.friedmanchisquare(*wide.to_numpy().T).pvalue,
                         "cd": NEMENYI[k] * np.sqrt(k * (k + 1) / (6 * n)),
                         **{f"rank:{s}": r for s, r in ranks.mean().sort_values().items()}})
    return pd.DataFrame(rows)


def pvalue(p):
    stars = "^{**}" if p < 0.01 else "^{*}" if p < 0.05 else ""
    return f"${'<.0001' if p < 1e-4 else f'{p:.4f}'.lstrip('0')}{stars}$"


def cells(row):
    return [f"{row['median']:+.2f}", f"{row['wins']}/{row['wins'] + row['losses']}", pvalue(row["p_holm"])]


def table(result, family, label, caption, first_column):
    """Rows are comparisons; columns are median delta, wins and Holm p for AUARC, then p-Brier."""
    chosen = result[result.family.str.startswith(family)]
    keys = chosen.drop_duplicates(["family", "protocol", "second"])
    lines = ["\\begin{table}[t]", "\\centering", "\\small", "\\setlength{\\tabcolsep}{3.5pt}",
             "\\begin{tabular}{ll rrc rrc}", "\\toprule",
             f" & & \\multicolumn{{3}}{{c}}{{AUARC}} & \\multicolumn{{3}}{{c}}{{p-Brier}} \\\\",
             "\\cmidrule(lr){3-5} \\cmidrule(lr){6-8}",
             f"{first_column} & vs. & Med.\\ $\\Delta$ & Wins & $p_{{\\mathrm{{Holm}}}}$ & Med.\\ $\\Delta$ & Wins & "
             "$p_{\\mathrm{Holm}}$ \\\\", "\\midrule"]
    previous = None
    for key in keys.itertuples():
        pair = chosen[(chosen.family == key.family) & (chosen.protocol == key.protocol) & (chosen.second == key.second)]
        values = [*cells(pair[pair.metric == "auarc"].iloc[0]), *cells(pair[pair.metric == "p_brier"].iloc[0])]
        head = first_label(key, family)
        if previous is not None and head != previous and head:
            lines.append("\\midrule")
        lines.append(f"{head if head != previous else ''} & {second_label(key, family)} & " + " & ".join(values) + " \\\\")
        previous = head
    lines += ["\\bottomrule", "\\end{tabular}", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\end{table}", ""]
    return "\n".join(lines)


def first_label(key, family):
    if family == "spaces":
        return "Consistency" if key.estimator == "consistency" else "Seq. Prob."
    return PROTOCOLS[key.protocol]


def second_label(key, family):
    if family == "spaces":
        return PROTOCOLS[key.protocol]
    if family == "interaction":
        return PROTOCOLS[key.second]
    return key.second


def ranking_table(ranks):
    lines = ["\\begin{table*}[t]", "\\centering", "\\small", "\\begin{tabular}{llrcp{0.44\\textwidth}}", "\\toprule",
             "Protocol & Metric & Friedman $p$ & CD & Average rank (best $\\to$ worst) \\\\", "\\midrule"]
    for _, row in ranks.iterrows():
        found = sorted((value, name.split(":", 1)[1]) for name, value in row.items()
                       if name.startswith("rank:") and pd.notna(value))
        chain = " $\\to$ ".join(f"{name} ({value:.2f})" for value, name in found)
        lines.append(f"{PROTOCOLS[row.protocol]} & {row.metric} & {pvalue(row.friedman_p)} & {row.cd:.2f} & {chain} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              "\\caption{\\textbf{Friedman ranks with the Nemenyi critical difference.} Within each of the 75 dataset "
              "$\\times$ group blocks, systems are ranked by the metric (1 = best); average ranks further apart than CD "
              "differ at $\\alpha = .05$.}", "\\label{tab:rankings}", "\\end{table*}", ""]
    return "\n".join(lines)


def main():
    frame = build()
    frame.to_csv(HERE / "blocks.csv", index=False)
    result, ranks = comparisons(frame), rankings(frame)
    result.to_csv(HERE / "significance.csv", index=False)
    ranks.to_csv(HERE / "rankings.csv", index=False)
    note = ("Each row compares log-odds Learned Weights with one system over the same 75 dataset $\\times$ group blocks; "
            "Med.\\ $\\Delta$ is in points and positive favors log-odds Learned Weights; Wins counts blocks it wins "
            "out of those that differ.")
    (HERE / "significance_baselines.tex").write_text(table(
        result, "baselines", "tab:significance",
        "\\textbf{Paired significance against the baselines (consistency).} " + note + " One-sided exact sign tests, "
        "Holm-corrected within each metric. $^{*}p<.05$, $^{**}p<.01$.", "Protocol"))
    (HERE / "significance_spaces.tex").write_text(table(
        result, "spaces", "tab:significance_spaces",
        "\\textbf{Log-odds against probability-space Learned Weights.} " + note + " Two-sided exact sign tests, "
        "Holm-corrected within each estimator and metric.", "Estimator"))
    (HERE / "significance_interaction.tex").write_text(table(
        result, "interaction", "tab:significance_interaction",
        "\\textbf{Does interaction enlarge the gain over the best agent?} Per block, the gain of log-odds Learned Weights "
        "over the group's best agent after a debate round minus the same gain before it. One-sided exact sign tests, "
        "Holm-corrected within each metric.", "Protocol"))
    (HERE / "rankings.tex").write_text(ranking_table(ranks))
    pd.set_option("display.width", 220)
    print(result[["family", "estimator", "protocol", "second", "metric", "median", "wins", "losses", "ties", "p_holm"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(ranks.round(3).T.to_string())


if __name__ == "__main__":
    main()
