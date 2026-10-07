import math
from statistics import mean


def holm_adjust(pvalues):
    ordered = sorted(range(len(pvalues)), key=pvalues.__getitem__)
    result = [1.0] * len(pvalues)
    previous = 0.0
    for position, index in enumerate(ordered):
        previous = max(previous, min(1.0, (len(pvalues) - position) * pvalues[index]))
        result[index] = previous
    return result


def sign_test(improvements):
    nonzero = [value for value in improvements if abs(value) > 1e-12]
    wins = sum(value > 0 for value in nonzero)
    n = len(nonzero)
    probability = sum(math.comb(n, count) for count in range(wins, n + 1)) / 2 ** n
    return wins, n, probability


def dataset_sign_tests(rows, cells, estimators, methods, tasks):
    results = []
    for estimator in estimators:
        for method in methods:
            if method == "best_solo":
                continue
            for metric in ("ece", "auarc"):
                differences, included = [], []
                for task in tasks:
                    expected = sum(cell["estimator"] == estimator and cell["task"] == task for cell in cells)
                    selected = [row for row in rows if row["estimator"] == estimator and row["task"] == task and row["method"] == method]
                    if expected and len(selected) == expected:
                        included.append(task)
                        differences.append(mean(row[f"delta_{metric}"] for row in selected) * (-1 if metric == "ece" else 1))
                wins, n, probability = sign_test(differences)
                results.append({"estimator": estimator, "method": method, "metric": metric,
                                "unit": "dataset_mean_over_panels", "alternative": "improves_over_reference",
                                "datasets": "|".join(included), "n_datasets": len(included),
                                "n_non_tied": n, "wins": wins, "p_raw": probability,
                                "p_holm": None, "exploratory": True})
    adjusted = holm_adjust([row["p_raw"] for row in results])
    for row, probability in zip(results, adjusted):
        row["p_holm"] = probability
    return results
