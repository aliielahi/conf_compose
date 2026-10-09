import ast
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from data import METRICS, atomic_module, index_judges, metric_values, reference_indices, restore_fit
from significance import holm_adjust, sign_test, dataset_sign_tests
from tables import aggregate, render_table, write_tables, write_significance_tables


def test_reference_is_metric_specific_and_uses_only_fitting_scores():
    fitting = [{"ece": 0.2, "auarc": 0.9}, {"ece": 0.1, "auarc": 0.8}]
    evaluation = [{"ece": 0.01, "auarc": 0.5}, {"ece": 0.7, "auarc": 0.99}]
    assert reference_indices("fit_metric", [0.7, 0.8], evaluation, [0.9, 0.95], fitting) == {"accuracy": 1, "ece": 1, "auarc": 0}
    assert reference_indices("metric_best", [0.7, 0.8], evaluation, [0.9, 0.95]) == {"accuracy": 1, "ece": 0, "auarc": 1}
    assert reference_indices("fit_accuracy", [0.9, 0.8], evaluation, [0.9, 0.95]) == dict.fromkeys(METRICS, 0)
    assert reference_indices("fit_auarc", [0.9, 0.8], evaluation, [0.9, 0.95], fitting, [0.4, 0.7]) == dict.fromkeys(METRICS, 1)
    assert reference_indices("fit_auarc", [0.9, 0.8], evaluation, [0.9, 0.95], fitting, [0.7, 0.7]) == dict.fromkeys(METRICS, 0)


def test_average_is_over_panel_deltas_with_missing_judges_explicit():
    cells = [{"estimator": "cons", "task": "csqa", "n_models": size} for size in (2, 2, 3)]
    rows = [{**cell, "method": "mean", "delta_ece": value, "delta_auarc": -value}
            for cell, value in zip(cells, (-0.1, -0.1, 0.2))]
    rows.append({**cells[0], "method": "judge", "delta_ece": 0.3, "delta_auarc": -0.3})
    result = aggregate(rows, cells, "cons", None, ["mean", "judge"], ["csqa"], True)
    assert result[("mean", "csqa", "ece")][0] == pytest.approx(0)
    assert result[("judge", "csqa", "ece")] == (0.3, 1, 3)
    restricted = aggregate(rows, cells, "cons", 2, ["mean", "judge"], ["csqa"], True)
    assert restricted[("mean", "csqa", "ece")] == (-0.1, 2, 2)
    text, latex = render_table(result, {"mean": "Arithmetic mean", "judge": "Judge"}, ["csqa"], "test", True)
    assert "[1/3]" not in text and "^{1/3}" not in latex
    assert "multicolumn{2}" in latex and "Delta ECE" in text
    assert latex.count("begin{tabular}") == latex.count("end{tabular}") == 1


def test_holm_adjustment_and_exact_sign_test():
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert sign_test([1, 2, 3]) == (3, 3, 0.125)
    assert sign_test([0, 0]) == (0, 0, 1)
    assert sign_test([-1, -2]) == (0, 2, 1)


def test_significance_uses_datasets_not_overlapping_panels():
    cells = [{"estimator": "cons", "task": task, "n_models": 2} for task in ("a", "b") for _ in range(15)]
    rows = [{**cell, "method": "mean", "delta_ece": -0.1, "delta_auarc": 0.1} for cell in cells]
    tests = dataset_sign_tests(rows, cells, ["cons"], ["mean"], ["a", "b"])
    assert all(row["n_datasets"] == row["n_non_tied"] == 2 for row in tests)
    assert all(row["p_raw"] == 0.25 and row["p_holm"] == 0.5 for row in tests)


def test_frozen_fits_are_reconstructed_without_optimizer():
    row = {"n_models": "2", "n_fit_scored": "20", "n_fit_errors": "5", "blp_scale": "",
           "blp_status": "fitted", "blp_parameters": json.dumps({"weights": [0.25, 0.75], "alpha": 1, "beta": 1})}
    with patch("conf_compose.composition.pooling.blp.minimize", side_effect=AssertionError("must not refit")):
        fit = restore_fit(row, "blp")
        assert fit.predict([0.2, 0.8]).score == pytest.approx(0.65)


def test_reporting_scripts_parse():
    for path in Path(__file__).parent.glob("*.py"):
        ast.parse(path.read_text())


def test_numerically_equal_confidences_remain_tied():
    result = metric_values([0.5, 0.5 + 1e-16], [0, 1])
    assert result["auarc"] == 0.5
    assert result["ece"] == 0


def test_missing_judge_directory_and_empty_results_fail(tmp_path):
    with pytest.raises(FileNotFoundError, match="--judge-dir"):
        index_judges(tmp_path / "missing", ["g3-27i"])
    with pytest.raises(ValueError, match="no completed judge cells"):
        index_judges(tmp_path, ["g3-27i"])


def test_coverage_is_separate_from_metric_tables(tmp_path):
    cells = [{"estimator": "cons", "task": "csqa", "n_models": 2} for _ in range(2)]
    rows = [{**cells[0], "method": "judge", "ece": 0.1, "auarc": 0.8, "delta_ece": 0.02, "delta_auarc": 0.03}]
    paths = write_tables(tmp_path, rows, cells, ["cons"], {"judge": "Judge"}, ["csqa"])
    assert "1/2" in (tmp_path / "cons/all_coverage.txt").read_text()
    assert "1/2" not in (tmp_path / "cons/all_delta.tex").read_text()
    assert len(paths) == 84


def test_significance_tables_keep_raw_and_adjusted_probabilities(tmp_path):
    (tmp_path / "cons").mkdir()
    tests = [{"estimator": "cons", "method": "mean", "metric": metric, "wins": 5,
              "n_non_tied": 5, "p_raw": 0.03125, "p_holm": 1.0} for metric in ("ece", "auarc")]
    write_significance_tables(tmp_path, tests, ["cons"], {"mean": "Mean"})
    text = (tmp_path / "cons/significance.txt").read_text()
    assert "5/5" in text and "0.0312" in text and "1.0000" in text


def test_selection_uses_consistency_for_both_families_and_preserves_majorities():
    from types import SimpleNamespace
    from conf_compose.composition.evidence import Item, Stream

    task = SimpleNamespace(name="test", equivalent=lambda a, b: a == b)
    streams = [Stream("a", 0, 0, "model-a", "A", ["A", "B", "B", "B", "B"]),
               Stream("b", 1, 0, "model-b", "B", ["B"] * 5)]
    item = Item("question", "q", "A", streams)
    records = {stream.model: {"question": {}} for stream in streams}
    args = SimpleNamespace(samples=5, estimator="cons", tie_break="confidence", tie_seed=0)
    atomic = atomic_module()
    assert atomic.confidence_scores(task, item, args, records)[0] == "B"
    args.estimator = "seq"
    with patch.object(atomic, "sequence_score", return_value=0.4):
        assert atomic.confidence_scores(task, item, args, records)[0] == "B"
    args.tie_break = "first"
    assert atomic.confidence_scores(task, item, args, records)[0] == "A"
    args.estimator, args.tie_break = "cons", "confidence"
    streams.append(Stream("c", 2, 0, "model-c", "A", ["B"] * 5))
    assert atomic.confidence_scores(task, item, args, records)[0] == "A"


def test_exact_confidence_ties_do_not_depend_on_cli_order():
    from types import SimpleNamespace
    from conf_compose.composition.evidence import Stream
    from conf_compose.composition.methods import majority_answer

    task = SimpleNamespace(equivalent=lambda a, b: a == b)
    streams = [Stream("a", 0, 0, "a", "A", []), Stream("b", 1, 0, "b", "B", [])]
    expected = majority_answer(task, streams, {"a": 0.5, "b": 0.5}, tie_key="question:0")
    reversed_streams = [Stream("b", 0, 0, "b", "B", []), Stream("a", 1, 0, "a", "A", [])]
    assert majority_answer(task, reversed_streams, {"a": 0.5, "b": 0.5}, tie_key="question:0") == expected


def test_full_metrics_include_unscaled_nll_and_accuracy_delta():
    from tables import FULL_METRICS

    cell = {"estimator": "cons", "task": "csqa", "n_models": 2}
    row = {**cell, "method": "mean", **{f"delta_{metric}": 0.12 for metric in FULL_METRICS}}
    values = aggregate([row], [cell], "cons", None, ["mean"], ["csqa"], True, FULL_METRICS)
    text, latex = render_table(values, {"mean": "Mean"}, ["csqa"], "test", True, FULL_METRICS)
    assert "+0.120" in text and "+12.00" in text
    assert f"multicolumn{{{len(FULL_METRICS)}}}" in latex and "AUROC" in latex and "p-ECE" in latex
    metrics = metric_values([0.2, 0.8], [0, 1])
    assert metrics["accuracy"] == 0.5 and metrics["auroc"] == 1
    assert metrics["brier"] == pytest.approx(0.04)
    assert metric_values([0.5], [1])["auroc"] is None


def test_missing_consistency_uses_seeded_fallback_and_records_reason():
    from types import SimpleNamespace
    from conf_compose.composition.evidence import Item, Stream

    task = SimpleNamespace(name="test", equivalent=lambda a, b: a == b)
    item = Item("question", "q", "A", [Stream("a", 0, 0, "a", "A", [None] * 5),
                                        Stream("b", 1, 0, "b", "B", ["B"] * 5)])
    args = SimpleNamespace(samples=5, tie_break="confidence", tie_seed=0)
    target, reason = atomic_module().selected_answer(task, item, args)
    assert target in ("A", "B") and reason == "missing_consistency_seeded"
