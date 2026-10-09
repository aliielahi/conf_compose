import csv
from pathlib import Path
from statistics import mean, stdev

METRICS = ("ece", "auarc")
FULL_METRICS = ("accuracy", "ece", "t_ece", "auarc", "answer_matched_auarc", "auroc", "brier", "t_brier", "nll")
METRIC_NAMES = {"accuracy": "Acc", "ece": "ECE", "t_ece": "t-ECE", "auarc": "AUARC",
                "answer_matched_auarc": "AUARC (same answer)", "auroc": "AUROC", "brier": "Brier", "t_brier": "t-Brier", "nll": "NLL"}
LOWER_IS_BETTER = {"ece", "t_ece", "brier", "t_brier", "nll"}
DATASET_NAMES = {"csqa": "CSQA", "boolq": "BoolQ", "gsm8k": "GSM8K", "truthfulqa": "TruthfulQA", "gpqa": "GPQA"}


def aggregate(rows, cells, estimator, size, methods, tasks, delta, metrics=METRICS, spread=False):
    values = {}
    for task in tasks:
        expected = sum(cell["estimator"] == estimator and cell["task"] == task
                       and (size is None or cell["n_models"] == size) for cell in cells)
        for method in methods:
            selected = [row for row in rows if row["estimator"] == estimator and row["task"] == task
                        and row["method"] == method and (size is None or row["n_models"] == size)]
            for metric in metrics:
                key = f"delta_{metric}" if delta else metric
                observed = [row[key] for row in selected if row.get(key) is not None]
                summary = (mean(observed) if observed else None, len(observed), expected)
                if spread:
                    summary += (stdev(observed) if len(observed) >= 2 else None,)
                values[(method, task, metric)] = summary
    return values


def escape(value):
    return str(value).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def display(value, available, expected, sd=None, *, delta, latex=False, metric="ece"):
    if value is None:
        return "--"
    scale, digits = (1, 3) if metric == "nll" else (100, 2)
    shown = round(scale * value, digits)
    shown = 0.0 if shown == 0 else shown
    text = f"{shown:+.{digits}f}" if delta else f"{shown:.{digits}f}"
    if delta and sd is not None:
        deviation = f"{scale * sd:.{digits}f}"
        text += (r" {\scriptsize $\pm$ " + deviation + "}") if latex else " ± " + deviation
    return text


def render_table(values, methods, tasks, title, delta, metrics=METRICS):
    metric_labels = [("Delta " if delta else "") + METRIC_NAMES[metric] for metric in metrics]
    headers = ["Method"] + [f"{DATASET_NAMES.get(task, task)} {metric}" for task in tasks for metric in metric_labels]
    rows = [[label] + [display(*values[(method, task, metric)], delta=delta, metric=metric)
                      for task in tasks for metric in metrics] for method, label in methods.items()]
    widths = [max(len(str(row[index])) for row in [headers, *rows]) for index in range(len(headers))]
    formatted = [" | ".join(value.ljust(width) for value, width in zip(row, widths)) for row in [headers, *rows]]
    formatted.insert(1, "-+-".join("-" * width for width in widths))
    notes = [title, "Acc, ECE, t-ECE, AUARC, AUROC, Brier and t-Brier are x100; NLL is unscaled in nats.",
             "Each model group has equal weight; the overall table averages groups, not size-table averages.",
             "Delta = method minus its own group's reference before averaging: negative ECE/Brier/NLL and positive Acc/AUARC/AUROC are improvements.",
             "Panel counts are in the matching coverage table; -- means unavailable. Partial rows use different groups.",
             "All present methods and solo models use the same scored questions within each group and estimator.",
             "Judge rows marked approximate reuse old scores, including changed targets; rerun before publication.", ""]
    if any(method.startswith("cagecal_") for method in methods):
        notes.insert(-1, "CAGE-CAL uses the IID voting adaptation and ensemble scores across training seeds; Dataset-specific BetaSB identity fallbacks are recorded in manifest.json.")
    spread = delta and any(len(value) == 4 for value in values.values())
    if spread:
        notes.insert(-1, "Mean ± sample standard deviation of group-level deltas (ddof=1); descriptive variation, not a standard error or confidence interval. SD omitted for fewer than two groups.")
    text = "\n".join(notes + formatted) + "\n"
    latex = [r"\begin{tabular}{l" + "r" * (len(metrics) * len(tasks)) + "}", r"\toprule",
             "Method & " + " & ".join(rf"\multicolumn{{{len(metrics)}}}{{c}}{{" + escape(DATASET_NAMES.get(task, task)) + "}" for task in tasks) + r" \\"]
    labels = [(r"$\Delta$ " if delta else "") + METRIC_NAMES[metric]
              + (r" $\downarrow$" if metric in LOWER_IS_BETTER else r" $\uparrow$") for metric in metrics]
    latex.append(" & " + " & ".join(labels * len(tasks)) + r" \\")
    latex.append(r"\midrule")
    for method, label in methods.items():
        numbers = [display(*values[(method, task, metric)], delta=delta, latex=True, metric=metric) for task in tasks for metric in metrics]
        latex.append(escape(label) + " & " + " & ".join(numbers) + r" \\")
    latex.append(r"\bottomrule")
    if spread:
        latex.append(r"\multicolumn{" + str(1 + len(tasks) * len(metrics))
                     + r"}{l}{\scriptsize Mean $\pm$ SD across groups (sample SD); descriptive, not a confidence interval.} \\")
    latex.extend([r"\end{tabular}", ""])
    if any("approx." in label for label in methods.values()):
        latex.insert(0, "% Approximate judge rows reuse scores elicited for old targets; regenerate before publication.")
    if any(method.startswith("cagecal_") for method in methods):
        latex.insert(0, "% CAGE-CAL: IID adaptation; seed-ensemble predictions. See manifest.json for dataset-specific BetaSB identity fallbacks.")
    latex.insert(0, "% Units: x100 except NLL (nats). Deltas are calculated within groups before averaging.")
    return text, "\n".join(latex)


def write_tables(directory, rows, cells, estimators, methods, tasks):
    paths = []
    for estimator in estimators:
        target = directory / estimator
        target.mkdir(parents=True, exist_ok=True)
        for size in (None, 2, 3, 4, 5, 6):
            scope = "all" if size is None else f"size_{size}"
            for tag, metrics in (("", METRICS), ("_full", FULL_METRICS), ("_accuracy", ("accuracy",))):
                for delta in (True, False):
                    kind = "delta" if delta else "absolute"
                    title = f"Voting protocol / {estimator} / {scope} / {kind}"
                    values = aggregate(rows, cells, estimator, size, methods, tasks, delta, metrics, spread=delta)
                    text, latex = render_table(values, methods, tasks, title, delta, metrics)
                    for suffix, content in (("txt", text), ("tex", latex)):
                        path = target / f"{scope}{tag}_{kind}.{suffix}"
                        path.write_text(content)
                        paths.append(str(path))
            coverage = aggregate(rows, cells, estimator, size, methods, tasks, True)
            headers = ["Method"] + [DATASET_NAMES.get(task, task) for task in tasks]
            entries = [[label] + [f"{coverage[(method, task, 'ece')][1]}/{coverage[(method, task, 'ece')][2]}"
                                  for task in tasks] for method, label in methods.items()]
            paths.extend(write_simple_table(target / f"{scope}_coverage", headers, entries,
                                            "Available/expected model groups; per-question coverage is in atomic.csv and audit.json."))
    return paths


def write_simple_table(stem, headers, entries, title):
    widths = [max(len(str(row[index])) for row in [headers, *entries]) for index in range(len(headers))]
    lines = [" | ".join(str(value).ljust(width) for value, width in zip(row, widths)) for row in [headers, *entries]]
    lines.insert(1, "-+-".join("-" * width for width in widths))
    text = title + "\n\n" + "\n".join(lines) + "\n"
    latex = [r"\begin{tabular}{l" + "r" * (len(headers) - 1) + "}", r"\toprule",
             " & ".join(escape(value) for value in headers) + r" \\", r"\midrule"]
    latex.extend(" & ".join(escape(value) for value in row) + r" \\" for row in entries)
    latex.extend([r"\bottomrule", r"\end{tabular}", ""])
    paths = []
    for suffix, content in (("txt", text), ("tex", "\n".join(latex))):
        path = stem.with_suffix(f".{suffix}")
        path.write_text(content)
        paths.append(str(path))
    return paths


def write_significance_tables(directory, tests, estimators, methods):
    paths = []
    for estimator in estimators:
        selected = {(row["method"], row["metric"]): row for row in tests if row["estimator"] == estimator}
        entries = []
        for method, label in methods.items():
            if method == "best_solo":
                continue
            entry = [label]
            for metric in METRICS:
                row = selected[(method, metric)]
                entry.extend([f"{row['wins']}/{row['n_non_tied']}", f"{row['p_raw']:.4f}", f"{row['p_holm']:.4f}"])
            entries.append(entry)
        headers = ["Method", "ECE wins/N", "ECE p", "ECE p Holm", "AUARC wins/N", "AUARC p", "AUARC p Holm"]
        paths.extend(write_simple_table(directory / estimator / "significance", headers, entries,
                                        "Exploratory one-sided dataset-block sign tests; N excludes ties. Holm covers both estimators and metrics jointly."))
    return paths


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paper_highlights(values, names, tasks, fields):
    highlights = {}
    macros = ("gfirst", "gsecond", "gthird")
    for task in tasks:
        for field in fields:
            candidates = sorted(
                ((name, values[name, task, field][0]) for name in names
                 if values[name, task, field][0] is not None),
                key=lambda item: item[1], reverse=field in ("auarc", "answer_matched_auarc", "delta_answer_matched_auarc"),
            )
            rank = 0
            previous = None
            for name, value in candidates:
                if previous is None or abs(value - previous) > 1e-12:
                    rank += 1
                    previous = value
                if rank > len(macros):
                    break
                highlights[name, task, field] = macros[rank - 1]
    return highlights


def write_paper_table(path, rows, cells, estimator, methods, tasks):
    sections = (
        ("Reference", ("best_solo",)),
        ("No learned combination weights", ("mean", "logodds_sum")),
        ("External baseline", ("cagecal_iid_betasb", "cagecal_iid")),
        ("Judge-based confidence", tuple(name for name in methods if name.startswith("judge:"))),
        ("Learned per-model weights", ("kahn", "kahn_diagonal", "blp", "logistic_pool")),
    )
    fields = ("t_ece", "t_brier", "auarc", "delta_answer_matched_auarc")
    values = aggregate(rows, cells, estimator, None, methods, tasks, False, fields, spread=True)
    names = [name for _, section in sections for name in section if name in methods]
    highlights = paper_highlights(values, names, tasks, fields)
    width = 1 + len(tasks) * len(fields)
    lines = [r"\begin{table*}[t]", r"\centering",
             rf"\caption{{Voting confidence quality with {escape(estimator)} scores. Absolute metrics are averaged across 15 model groups per dataset; entries are mean $\pm$ sample SD. Lower t-ECE and t-Brier, and higher AUARC, are better.}}",
             rf"\label{{tab:voting-{escape(estimator)}-absolute}}",
             r"\setlength{\tabcolsep}{2.6pt}", r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{@{}l" + "r" * (width - 1) + r"@{}}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{4}}{{c}}{{{escape(DATASET_NAMES.get(task, task))}}}" for task in tasks) + r" \\",
             "Method & " + " & ".join(("t-ECE $\\downarrow$", "t-Brier $\\downarrow$", "AUARC $\\uparrow$", "$\\Delta$AUARC (same answer) $\\uparrow$") * len(tasks)) + r" \\",
             r"\midrule"]
    for section, names in sections:
        selected = [name for name in names if name in methods]
        if not selected:
            continue
        lines.append(rf"\multicolumn{{{width}}}{{l}}{{\textbf{{{section}}}}}\\")
        for name in selected:
            cells_out = []
            for task in tasks:
                for field in fields:
                    value, _, _, deviation = values[name, task, field]
                    cell = "--" if value is None else f"{value:.3f}" + (
                        rf" {{\scriptsize $\pm$ {deviation:.3f}}}" if deviation is not None else ""
                    )
                    macro = highlights.get((name, task, field))
                    cells_out.append(rf"\{macro}{{{cell}}}" if macro else cell)
            lines.append(escape(methods[name]) + " & " + " & ".join(cells_out) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}%", "}",
              r"\vspace{2pt}",
              r"\parbox{\textwidth}{\footnotesize t-ECE and t-Brier use one output temperature fitted by NLL on fitting data. AUARC uses original scores; $\Delta$AUARC compares with the same selected answer scored by one fitting-accuracy-selected model. CAGE-CAL BetaSB has no additional temperature fit.}",
              r"\end{table*}"]
    path.write_text("\n".join(lines) + "\n")
