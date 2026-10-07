import csv
from pathlib import Path
from statistics import mean

METRICS = ("ece", "auarc")
FULL_METRICS = ("accuracy", "ece", "auarc", "auroc", "brier", "nll")
METRIC_NAMES = {"accuracy": "Acc", "ece": "ECE", "auarc": "AUARC", "auroc": "AUROC", "brier": "Brier", "nll": "NLL"}
LOWER_IS_BETTER = {"ece", "brier", "nll"}
DATASET_NAMES = {"csqa": "CSQA", "boolq": "BoolQ", "gsm8k": "GSM8K", "truthfulqa": "TruthfulQA", "gpqa": "GPQA"}


def aggregate(rows, cells, estimator, size, methods, tasks, delta, metrics=METRICS):
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
                values[(method, task, metric)] = (mean(observed) if observed else None, len(observed), expected)
    return values


def escape(value):
    return str(value).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def display(value, available, expected, delta, latex=False, metric="ece"):
    if value is None:
        return "--"
    scale, digits = (1, 3) if metric == "nll" else (100, 2)
    shown = round(scale * value, digits)
    shown = 0.0 if shown == 0 else shown
    text = f"{shown:+.{digits}f}" if delta else f"{shown:.{digits}f}"
    return text


def render_table(values, methods, tasks, title, delta, metrics=METRICS):
    metric_labels = [("Delta " if delta else "") + METRIC_NAMES[metric] for metric in metrics]
    headers = ["Method"] + [f"{DATASET_NAMES.get(task, task)} {metric}" for task in tasks for metric in metric_labels]
    rows = [[label] + [display(*values[(method, task, metric)], delta, metric=metric)
                      for task in tasks for metric in metrics] for method, label in methods.items()]
    widths = [max(len(str(row[index])) for row in [headers, *rows]) for index in range(len(headers))]
    formatted = [" | ".join(value.ljust(width) for value, width in zip(row, widths)) for row in [headers, *rows]]
    formatted.insert(1, "-+-".join("-" * width for width in widths))
    notes = [title, "Acc, ECE, AUARC, AUROC and Brier are x100; NLL is unscaled in nats.",
             "Each model group has equal weight; the overall table averages groups, not size-table averages.",
             "Delta = method minus its own group's reference before averaging: negative ECE/Brier/NLL and positive Acc/AUARC/AUROC are improvements.",
             "Panel counts are in the matching coverage table; -- means unavailable. Partial rows use different groups.",
             "All present methods and solo models use the same scored questions within each group and estimator.",
             "Judge rows marked approximate reuse old scores, including changed targets; rerun before publication.", ""]
    text = "\n".join(notes + formatted) + "\n"
    latex = [r"\begin{tabular}{l" + "r" * (len(metrics) * len(tasks)) + "}", r"\toprule",
             "Method & " + " & ".join(rf"\multicolumn{{{len(metrics)}}}{{c}}{{" + escape(DATASET_NAMES.get(task, task)) + "}" for task in tasks) + r" \\"]
    labels = [(r"$\Delta$ " if delta else "") + METRIC_NAMES[metric]
              + (r" $\downarrow$" if metric in LOWER_IS_BETTER else r" $\uparrow$") for metric in metrics]
    latex.append(" & " + " & ".join(labels * len(tasks)) + r" \\")
    latex.append(r"\midrule")
    for method, label in methods.items():
        numbers = [display(*values[(method, task, metric)], delta, latex=True, metric=metric) for task in tasks for metric in metrics]
        latex.append(escape(label) + " & " + " & ".join(numbers) + r" \\")
    latex.extend([r"\bottomrule", r"\end{tabular}", ""])
    if any("approx." in label for label in methods.values()):
        latex.insert(0, "% Approximate judge rows reuse scores elicited for old targets; regenerate before publication.")
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
                    values = aggregate(rows, cells, estimator, size, methods, tasks, delta, metrics)
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
