import csv
import importlib.util
import json
import math
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from conf_compose.composition.evidence import Item, Stream
from conf_compose.composition.pooling import fit_shared_rho, fit_shared_scale, pool_methods
from conf_compose.utils.metrics import nll


def probabilities(logits):
    return 1 / (1 + np.exp(-np.asarray(logits)))


def test_shared_rho_removes_class_means_before_estimating_dependence():
    residuals = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    logits = np.vstack([residuals - 3, residuals + 3])
    labels = [0] * 4 + [1] * 4
    fit = fit_shared_rho(probabilities(logits), labels)
    assert np.corrcoef(logits.T)[0, 1] > 0.8
    assert fit.parameters["rho"] == pytest.approx(0, abs=1e-12)
    assert fit.scale == pytest.approx(1)


def test_shared_rho_discounts_exact_duplicates_and_preserves_stream_order_invariance():
    logits = np.array([-2, -1, 0, 1, 2, 3])
    scores = probabilities(np.column_stack([logits] * 3))
    fit = fit_shared_rho(scores, [0, 0, 0, 1, 1, 1])
    assert fit.parameters["rho"] == pytest.approx(1)
    assert fit.scale == pytest.approx(1 / 3)
    assert fit.predict([0.8] * 3).score == pytest.approx(0.8)
    assert fit_shared_rho(scores[:, ::-1], [0, 0, 0, 1, 1, 1]).scale == pytest.approx(fit.scale)


def test_shared_rho_negative_correlation_is_recorded_and_clipped():
    logits = np.array([[-2, 2], [-1, 1], [1, -1], [2, -2]])
    fit = fit_shared_rho(probabilities(logits), [0, 0, 1, 1])
    assert fit.parameters["raw_rho"] == pytest.approx(-1)
    assert fit.parameters["rho"] == 0
    assert fit.scale == 1


def test_degenerate_fits_are_explicit_and_single_stream_rho_is_identity():
    assert fit_shared_rho([[0.7, 0.7]] * 5, [1] * 5).status == "insufficient_classes"
    fit = fit_shared_rho([[0.7, 0.7]] * 4, [0, 0, 1, 1])
    assert fit.status == "zero_variance"
    assert fit.predict([0.7, 0.7]).score is None
    assert fit_shared_rho([[0.7]], [1]).predict([0.7]).score == pytest.approx(0.7)
    assert fit_shared_scale([[0.7, 0.7]] * 4, [1] * 4).status == "insufficient_classes"
    assert fit_shared_scale([[0.5, 0.5]] * 4, [0, 0, 1, 1]).status == "zero_information"
    for fitter in (fit_shared_rho, fit_shared_scale):
        assert fitter(np.empty((0, 2)), []).status == "insufficient_classes"


def test_shared_scale_recovers_known_nll_optimum():
    labels = [1] * 8 + [0] * 2
    fit = fit_shared_scale([[0.9, 0.9]] * 10, labels)
    expected = math.log(4) / (2 * math.log(9))
    assert fit.scale == pytest.approx(expected)
    assert fit.predict([0.9, 0.9]).score == pytest.approx(0.8)
    assert nll([0.8] * 10, labels) < nll([pool_methods([0.9, 0.9])["logodds_sum"].score] * 10, labels)
    assert fit_shared_scale([[0.6]] * 10, [1] * 9 + [0]).scale > 1


def test_shared_scale_boundary_solutions_and_validation():
    assert fit_shared_scale([[0.1], [0.9]], [1, 0]).status == "lower_bound"
    assert fit_shared_scale([[0.1], [0.9]], [0, 1]).status == "upper_bound"
    for fitter in (fit_shared_rho, fit_shared_scale):
        for scores, labels in (([[float("nan")]], [1]), ([[1.1]], [0]), ([[0.5]], [2]), ([[0.5]], [])):
            with pytest.raises(ValueError):
                fitter(scores, labels)
    fit = fit_shared_scale([[0.8], [0.2]], [1, 0])
    with pytest.raises(ValueError):
        fit.predict([0.2, 0.3])


@pytest.fixture
def atomic():
    path = Path(__file__).resolve().parents[1] / "runs/experiment02-voting_composition/atomic.py"
    spec = importlib.util.spec_from_file_location("atomic_learned", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def arguments(**overrides):
    values = dict(estimator="cons", context="direct", seq_score="norm_sum", match="test", voter=0,
                  samples=5, fit_fraction=0.3, fit_seed=0, fit_split="holdout")
    return SimpleNamespace(**(values | overrides))


def example_items(prefix, count=80):
    result = []
    for index in range(count):
        streams = [Stream(f"{prefix}{index}:a{agent}", agent, 0, f"model{agent}", "A",
                          ["A"] * ((index + agent) % 5 + 1) + ["B"] * (4 - (index + agent) % 5))
                   for agent in range(3)]
        result.append(Item(f"{prefix}{index}", "question", "B" if index % 3 == 0 else "A", streams))
    return result


def test_split_is_stable_across_order_and_panels(atomic):
    items = example_items("e")
    fitting, evaluation = atomic.split_items("csqa", items, 0.3, 0)
    reversed_fit, _ = atomic.split_items("csqa", list(reversed(items)), 0.3, 0)
    reduced_fit, _ = atomic.split_items("csqa", items[:60], 0.3, 0)
    assert {item.example_id for item in fitting}.isdisjoint(item.example_id for item in evaluation)
    assert {item.example_id for item in fitting} == {item.example_id for item in reversed_fit}
    assert {item.example_id for item in reduced_fit} == {item.example_id for item in fitting if item in items[:60]}


def test_atomic_fitting_cannot_see_evaluation_labels_and_ranking_is_preserved(atomic):
    fitting, evaluation = example_items("fit"), example_items("test")
    args = arguments(fit_split="validation")
    original = atomic.run_task("csqa", args, evaluation, {}, fitting)
    altered = atomic.run_task("csqa", args, [replace(item, gold="B") for item in evaluation], {}, fitting)
    for name in ("shared_rho", "shared_scale"):
        assert original[f"{name}_scale"] == altered[f"{name}_scale"]
        assert original[f"{name}_status"] == "fitted"
        assert original[f"{name}_auroc"] == original["logodds_sum_auroc"]
        assert original[f"{name}_auarc"] == original["logodds_sum_auarc"]
        assert original[f"{name}_coverage"] == 1
    for name in ("kahn", "blp", "logistic_pool"):
        assert original[f"{name}_parameters"] == altered[f"{name}_parameters"]
        assert original[f"{name}_status"] == altered[f"{name}_status"] == "fitted"
    assert original["kahn_status"] == altered["kahn_status"] == "fitted"
    assert original["single_acc"] != altered["single_acc"]
    with pytest.raises(ValueError, match="overlap"):
        atomic.run_task("csqa", args, evaluation, {}, evaluation)


def test_missing_scores_reduce_all_pool_coverage_together(atomic):
    fitting, evaluation = example_items("fit"), example_items("test", 20)
    evaluation[0].streams[0].samples = []
    row = atomic.run_task("csqa", arguments(fit_split="validation"), evaluation, {}, fitting)
    for name in atomic.RULES:
        assert row[f"{name}_coverage"] == 0.95
    assert row["vote_acc"] == pytest.approx(13 / 20, abs=1e-4)


def test_sequence_scores_use_the_shared_target(atomic):
    records = {"candidates": [{"answer": "A", "direct": {"logprobs": [math.log(0.8)]}},
                              {"answer": "B", "direct": {"logprobs": [math.log(0.2)]}}]}
    task = atomic.get_task("csqa")
    assert atomic.sequence_score(task, {"candidate_scores": records}, "B", arguments()) == pytest.approx(0.2)


@pytest.mark.parametrize("estimator", ["cons", "seq"])
def test_atomic_cli_generates_all_methods_without_modifying_inference(tmp_path, atomic, estimator):
    store = tmp_path / "inference"
    for agent in range(3):
        cell = store / "csqa" / f"model{agent}--s70v0_k5_fixture"
        cell.mkdir(parents=True)
        for split, prefix in (("test", "test"), ("validation", "fit")):
            rows = []
            for index, item in enumerate(example_items(prefix)):
                p = 0.2 + 0.12 * ((index + agent) % 5)
                rows.append({"id": item.example_id, "question": item.question, "gold": item.gold,
                             "prediction": "A", "sampled_answers": {"consistency_t0.7": item.streams[agent].samples},
                             "candidate_scores": {"candidates": [
                                 {"answer": "A", "direct": {"logprobs": [math.log(p)]}},
                                 {"answer": "B", "direct": {"logprobs": [math.log(1-p)]}}]}})
            (cell / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    before = {path: path.read_bytes() for path in store.rglob("*.jsonl")}
    output = tmp_path / "rows.csv"
    command = [sys.executable, atomic.__file__, "--tasks", "csqa", "--group", "model0", "model1", "model2",
               "--sizes", "2", "3", "--store", str(store), "--match", "fixture", "--out", str(output),
               "--estimator", estimator, "--fit-split", "validation"]
    result = subprocess.run(command, capture_output=True, text=True, env=os.environ)
    assert result.returncode == 0, result.stderr
    rows = list(csv.DictReader(output.open()))
    assert len(rows) == 4
    assert rows == list(csv.DictReader(result.stdout.splitlines()))
    for row in rows:
        assert row["n_examples"] == row["n_fit_questions"] == "80"
        assert row["fit_ids_hash"] != row["eval_ids_hash"]
        assert row["shared_rho_status"] == "fitted"
        assert row["kahn_status"] == "fitted"
        assert len(json.loads(row["kahn_parameters"])["weights"]) == int(row["n_models"])
        for rule in atomic.RULES:
            assert row[f"{rule}_ece"] != ""
    assert before == {path: path.read_bytes() for path in store.rglob("*.jsonl")}
    output.write_text("old,columns\n")
    result = subprocess.run(command, capture_output=True, text=True, env=os.environ)
    assert result.returncode != 0
    assert "CSV columns differ" in result.stderr
    assert output.read_text() == "old,columns\n"


def test_ablations_use_the_same_questions_and_add_only_comparison_columns(atomic):
    fitting, evaluation = example_items("fit"), example_items("test", 20)
    base = atomic.run_task("csqa", arguments(fit_split="validation"), evaluation, {}, fitting)
    controls = atomic.run_task("csqa", arguments(fit_split="validation", ablations=True), evaluation, {}, fitting)
    assert all(controls[key] == value for key, value in base.items())
    assert controls["blp_equal_auroc"] == controls["mean_auroc"]
    assert controls["blp_equal_auarc"] == controls["mean_auarc"]
    for name in ("blp_equal", "kahn_diagonal"):
        assert controls[f"{name}_coverage"] == 1
        assert controls[f"{name}_status"] == "fitted"


def test_panel_sweep_writes_each_exact_group_and_estimator(tmp_path, atomic):
    from conf_compose.constants import COMPOSITION_PANELS
    models = list(dict.fromkeys(model for panel in COMPOSITION_PANELS for model in panel))
    store = tmp_path / "inference"
    for agent, model in enumerate(models):
        cell = store / "csqa" / f"vllm__{model}--s70v0_k5_fixture"
        cell.mkdir(parents=True)
        for split, prefix in (("test", "test"), ("validation", "fit")):
            rows = []
            for index in range(30):
                probability = 0.2 + 0.12 * ((index + agent) % 5)
                rows.append({"id": f"{prefix}{index}", "question": "question", "gold": "B" if index % 3 == 0 else "A",
                             "prediction": "A", "sampled_answers": {"consistency_t0.7":
                                 ["A"] * ((index + agent) % 5 + 1) + ["B"] * (4 - (index + agent) % 5)},
                             "candidate_scores": {"candidates": [
                                 {"answer": "A", "direct": {"logprobs": [math.log(probability)]}},
                                 {"answer": "B", "direct": {"logprobs": [math.log(1 - probability)]}}]}})
            (cell / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    before = {path: path.read_bytes() for path in store.rglob("*.jsonl")}
    output = tmp_path / "out"
    command = [sys.executable, str(Path(atomic.__file__).with_name("sweep_learned.py")),
               "--tasks", "csqa", "--estimators", "cons", "seq", "--store", str(store),
               "--match", "fixture", "--fit-split", "validation", "--ablations", "--out-dir", str(output)]
    result = subprocess.run(command, capture_output=True, text=True, env=os.environ)
    assert result.returncode == 0, result.stderr
    for estimator in ("cons", "seq"):
        rows = list(csv.DictReader((output / f"{estimator}.csv").open()))
        assert [row["models"] for row in rows] == ["|".join(panel) for panel in COMPOSITION_PANELS]
        assert all(row["estimator"] == estimator and row["n_samples"] == "5" for row in rows)
        for row in rows:
            assert len(row["code_hash"]) == 64
            assert len(json.loads(row["input_hashes"])) == int(row["n_models"])
            for rule in (*atomic.RULES, "blp_equal", "kahn_diagonal"):
                assert row[f"{rule}_coverage"] == "1.0"
                assert row[f"{rule}_nll"] != ""
    outputs_before = {path: path.read_bytes() for path in output.iterdir()}
    result = subprocess.run(command, capture_output=True, text=True, env=os.environ)
    assert result.returncode != 0
    assert "already exist" in result.stderr
    assert outputs_before == {path: path.read_bytes() for path in output.iterdir()}
    assert before == {path: path.read_bytes() for path in store.rglob("*.jsonl")}
