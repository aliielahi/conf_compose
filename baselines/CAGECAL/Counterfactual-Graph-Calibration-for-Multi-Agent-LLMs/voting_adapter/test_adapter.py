"""Light CPU tests: python -B -m unittest voting_adapter.test_adapter -v.

Optional graph smoke: python -B -m voting_adapter.run --smoke --device cuda:0.
Synthetic files exist only in temporary directories inside this baseline.
"""
from __future__ import annotations

import copy
import csv
import importlib.util
import json
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path

import numpy as np

from .data import (ROOT, DEFAULT_PROJECT, fitting_id, id_digest, internal_splits,
                   majority, select_vote, pool_source, portable_source, file_hash, project_api, read_records, safe_output)
from .features import answer_projection, pearson_matrix, query_matrices


class FakeBeta:
    def __init__(self, **kwargs):
        pass

    def fit(self, p, y):
        return self

    def predict(self, p):
        return np.asarray(p).ravel()


class ClosurePlatt:
    def __init__(self, **kwargs):
        pass

    def train_calibration(self, p, y):
        # Reproduce the real package's unpickleable local-function state.
        self.calibrator = lambda values: 0.5 * np.asarray(values) + 0.25

    def calibrate(self, p):
        return self.calibrator(p)


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.get_task = staticmethod(project_api(DEFAULT_PROJECT))

    def test_task_equivalence_and_ordered_ties(self):
        task = self.get_task("gsm8k")
        target, answers = majority(task, ["12.0", "12", "15"])
        self.assertEqual(target, "12.0")
        self.assertEqual(answers, ["12.0", "12.0", "15"])
        self.assertEqual(majority(task, ["15", "12"])[0], "15")
        self.assertEqual(majority(self.get_task("boolq"), ["yes", "true", "no"])[0], "yes")

    def test_missing_answer_never_votes(self):
        task = self.get_task("csqa")
        self.assertEqual(majority(task, [None, "B", "A"])[0], "B")
        self.assertEqual(majority(task, [None, ""])[0], None)

    def test_confidence_tie_break_and_strict_majority(self):
        task = self.get_task("csqa")
        members = [dict(prediction="A", samples=["A"] + ["B"] * 4),
                   dict(prediction="B", samples=["B"] * 5),
                   dict(prediction="C", samples=["C"] * 3 + [None] * 2)]
        result = select_vote(task, members, ["a", "b", "c"], "q", tie_break="confidence")
        self.assertEqual((result[0], result[2]), ("B", "confidence"))
        self.assertEqual(select_vote(task, members, ["a", "b", "c"], "q")[0], "A")
        # The five original samples resolve ties only; they never overturn 2:1.
        members[2]["prediction"] = "A"
        self.assertEqual(select_vote(task, members, ["a", "b", "c"], "q", tie_break="confidence")[0], "A")

    def test_seeded_ties_are_order_independent_and_ignore_gold(self):
        task = self.get_task("csqa")
        members = [dict(prediction=a, samples=[a] * 5) for a in ("A", "B", "C")]
        target, _, reason = select_vote(task, members, ["a", "b", "c"], "q", tie_break="confidence")
        self.assertEqual(reason, "equal_confidence_seeded")
        reversed_target, _, _ = select_vote(task, members[::-1], ["c", "b", "a"], "q", tie_break="confidence")
        self.assertEqual(target, reversed_target)
        for member in members:
            member["gold"] = "wrong label"
        self.assertEqual(target, select_vote(task, members, ["a", "b", "c"], "q", tie_break="confidence")[0])
        members[0]["samples"] = [None] * 5
        self.assertEqual(select_vote(task, members, ["a", "b", "c"], "q", tie_break="confidence")[2],
                         "missing_consistency_seeded")

    def test_pool_manifest_relocation_and_hash_validation(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "voting_adapter", prefix="test-") as tmp:
            project = Path(tmp)
            pool = project / "results/voting_atomic/new_rule/cons.csv"
            pool.parent.mkdir(parents=True)
            pool.write_text("a,b\n1,2\n")
            saved = "/old/mac/conf_compose/results/voting_atomic/new_rule/cons.csv"
            manifest = {"pool_inputs": {saved: file_hash(pool)}}
            self.assertEqual(pool_source(project, manifest, "cons")[0], pool)
            pool.write_text("modified")
            with self.assertRaisesRegex(ValueError, "differs from the paper manifest"):
                pool_source(project, manifest, "cons")
            with self.assertRaises(ValueError):
                portable_source(project, "../outside.csv")

    def test_split_groups_by_question_not_panel(self):
        ids = {str(i) for i in range(100)}
        fit = {q for q in ids if fitting_id("boolq", q, 0.3, 0)}
        train, val = internal_splits("boolq", fit, 0)
        self.assertFalse(train & val)
        self.assertEqual(train | val, fit)
        self.assertEqual((train, val), internal_splits("boolq", reversed(sorted(fit)), 0))
        self.assertEqual(id_digest(fit), id_digest(reversed(sorted(fit))))

    def test_pca_never_fits_evaluation_answers(self):
        emb = {"a": np.array([1., 0, 0]), "b": np.array([0., 1, 0]), "eval": np.array([0., 0, 9.])}
        first, state = answer_projection(emb, ["a", "b", "a"])
        emb["eval"] *= 100
        second, changed = answer_projection(emb, ["a", "b"])
        np.testing.assert_array_equal(state["basis"], changed["basis"])
        np.testing.assert_array_equal(first["a"], second["a"])
        self.assertEqual(first["eval"].shape, (16,))
        self.assertTrue(np.all(first["a"][1:] == 0))
        one, _ = answer_projection(emb, ["a"])
        np.testing.assert_array_equal(one["a"], np.zeros(16))

    def test_correlations_ignore_validation_and_evaluation_labels(self):
        rows = [dict(task="t", models="a|b", model_ids=["a", "b"], id=str(i),
                     split="train" if i < 3 else "validation" if i == 3 else "evaluation",
                     member_correct=[i % 2, i % 2]) for i in range(5)]
        emb = {("t", str(i)): np.array([1., float(i)]) for i in range(5)}
        original = query_matrices(rows, emb, 20)
        changed = copy.deepcopy(rows)
        for row in changed[3:]:
            del row["member_correct"]  # The function must not read these labels at all.
        compared = query_matrices(changed, emb, 20)
        for key, (matrix, neighbors) in original.items():
            np.testing.assert_array_equal(matrix, compared[key][0])
            self.assertLessEqual(set(neighbors), {"0", "1", "2"})
            self.assertNotIn(key[2], neighbors)
        np.testing.assert_allclose(original[("t", "a|b", "4")][0], np.ones((2, 2)), atol=1e-6)

    def test_degenerate_correlations(self):
        np.testing.assert_array_equal(pearson_matrix([[1, 1], [1, 1]], 2), np.eye(2))
        np.testing.assert_array_equal(pearson_matrix([], 3), np.eye(3))

    def test_output_cannot_escape_baseline(self):
        with self.assertRaises(ValueError):
            safe_output(DEFAULT_PROJECT / "results")
        with self.assertRaises(ValueError):
            safe_output(ROOT / ".." / "outside")
        self.assertEqual(safe_output(ROOT / "results/test"), ROOT / "results/test")

    def test_record_reader_rejects_duplicate_ids_and_preserves_zero_lp(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "voting_adapter", prefix="test-") as tmp:
            p = Path(tmp) / "synthetic.jsonl"
            record = dict(id="q", prediction="A", token_logprobs={"response": [0.0, 0.0]})
            p.write_text(json.dumps(record) + "\n")
            self.assertEqual(read_records(p)["q"]["mean_logprob"], 0)
            p.write_text((json.dumps(record) + "\n") * 2)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                read_records(p)

    def test_report_exact_mask_and_reject_changed_target(self):
        from .report import make_reports
        with tempfile.TemporaryDirectory(dir=ROOT / "voting_adapter", prefix="test-") as tmp:
            tmp = Path(tmp)
            base = dict(task="csqa", estimator="cons", models="a|b", n_models=2, n_matched=2,
                coverage=2/3, reference_mode="fit_metric", method="mean", accuracy=0.5,
                ece=0.2, reference_ece=0.3, delta_ece=-0.1, reference_ece_model="a", reference_ece_accuracy=0.5,
                auarc=0.6, reference_auarc=0.5, delta_auarc=0.1, reference_auarc_model="a", reference_auarc_accuracy=0.5)
            for metric in ("accuracy", "auroc", "brier", "nll"):
                base.setdefault(metric, 0.25)
                base.update({"reference_" + metric: 0.2, "delta_" + metric: 0.05,
                             "reference_" + metric + "_model": "a", "reference_" + metric + "_accuracy": 0.5})
            source = tmp / "original.csv"
            with source.open("w") as f:
                writer = csv.DictWriter(f, fieldnames=list(base)); writer.writeheader()
                writer.writerow(dict(base, method="best_solo", accuracy=""))
                writer.writerow(base)
            rows = [dict(task="csqa", models="a|b", id=str(i), target="A", correct=int(i == 1), split="evaluation")
                    for i in range(3)]
            cell = dict(task="csqa", estimator="cons", models="a|b", n_models=2,
                        matched_ids=["1", "2"], coverage=2/3, vote_accuracy=0.5)
            bundle = dict(project=DEFAULT_PROJECT, rows=rows, cells=[cell], atomic_path=source)
            predictions = [dict(**{k: v for k, v in r.items() if k != "split"},
                                raw=[0.99, 0.9, 0.1][i], betasb=[0.99, 0.9, 0.1][i]) for i, r in enumerate(rows)]
            path = tmp / "synthetic_predictions.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in predictions))
            original_bytes = source.read_bytes()
            report = make_reports(bundle, path, tmp / "tables")
            self.assertEqual(source.read_bytes(), original_bytes)
            self.assertEqual(len(report), 2)
            self.assertEqual(report[0]["n"], 2)
            self.assertAlmostEqual(report[0]["ece"], 0.1)
            self.assertAlmostEqual(report[0]["auarc"], 0.75)
            self.assertEqual(report[0]["accuracy"], 0.5)
            self.assertTrue((tmp / "tables/cons/size_2_absolute.tex").exists())
            with (tmp / "tables/atomic.csv").open() as handle:
                cage = next(r for r in csv.DictReader(handle) if r["method"] == "cagecal_iid")
            self.assertAlmostEqual(float(cage["auroc"]), 1.0)
            self.assertAlmostEqual(float(cage["delta_auroc"]), 0.8)
            self.assertAlmostEqual(float(cage["brier"]), 0.01)
            self.assertAlmostEqual(float(cage["delta_accuracy"]), 0.3)
            predictions[0]["target"] = "B"
            path.write_text("".join(json.dumps(r) + "\n" for r in predictions))
            with self.assertRaisesRegex(ValueError, "changed fixed vote"):
                make_reports(bundle, path, tmp / "tables")

    def test_closure_calibration_and_prediction_only_recovery(self):
        from .training import finish_saved
        modules = {"betacal": SimpleNamespace(BetaCalibration=FakeBeta),
                   "calibration": SimpleNamespace(PlattBinnerCalibrator=ClosurePlatt)}
        validation = [dict(task="csqa", models="a|b", id=str(i), target="A", correct=i % 2,
                           split="validation") for i in range(60)]
        evaluation = [dict(task="csqa", models="a|b", id="test" + str(i), target="B", correct=i,
                           split="evaluation") for i in range(2)]
        bundle = dict(rows=validation + evaluation, splits={"csqa": {}})
        args = SimpleNamespace(seeds=1, split_seed=0)
        with tempfile.TemporaryDirectory(dir=ROOT / "voting_adapter", prefix="test-") as tmp, patch.dict("sys.modules", modules):
            tmp = Path(tmp)
            ledger = tmp / "features.jsonl"
            ledger.write_text("".join(json.dumps(r) + "\n" for r in bundle["rows"]))
            np.savez(tmp / "seed_0_predictions.npz", validation=np.linspace(0.1, 0.9, 60),
                     evaluation=np.array([0.3, 0.7]))
            original = (tmp / "seed_0_predictions.npz").read_bytes()
            path = finish_saved(bundle, args, tmp)
            predictions = [json.loads(line) for line in path.read_text().splitlines()]
            np.testing.assert_allclose([r["betasb"] for r in predictions], [0.35, 0.65])
            self.assertFalse((tmp / "calibrators.pkl").exists())
            self.assertEqual(original, (tmp / "seed_0_predictions.npz").read_bytes())
            with np.load(tmp / "calibration_inputs.npz", allow_pickle=False) as saved:
                self.assertEqual(saved["validation_ids"].shape, (60,))
                np.testing.assert_allclose(saved["calibrated_probabilities"], [0.35, 0.65])
            ledger.write_text("".join(json.dumps(r) + "\n" for r in bundle["rows"][::-1]))
            with self.assertRaisesRegex(ValueError, "order/targets differ"):
                finish_saved(bundle, args, tmp)

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("torch_geometric"),
                         "PyTorch/PyG not available; run --smoke in the GPU environment")
    def test_upstream_graph_forward_backward(self):
        graph_smoke("cpu")


def graph_smoke(device="cpu"):
    """Actual upstream model forward/backward, mixed-size batching and checkpoint reload."""
    import torch
    from .training import make_graph, collate, upstream
    torch.manual_seed(0)
    examples = []
    for n in (2, 3, 5):
        row = dict(model_ids=[f"model{i}" for i in range(n)], task="synthetic",
                   answers=["A" if i % 2 else "B" for i in range(n)], mean_logprobs=[0.] + [-1.] * (n-1),
                   correct=int(n % 2 == 0), target="A" if n == 2 else "B")
        features = {"A": np.ones(16, dtype=np.float32), "B": np.zeros(16, dtype=np.float32)}
        data = make_graph(row, np.eye(n, dtype=np.float32), features)
        assert data.x_T.shape == (n, 23)
        assert data.x_T[0, 0].item() == 1.0
        assert torch.equal(data.x_T, data.x_0)
        assert (data.edge_attr_T[:, 0] == 0).all()
        if n == 2:  # Fixed answer A wins the tie despite B being proposed first.
            torch.testing.assert_close(data.x_T[:, 5], torch.tensor([0., 1.]))
            torch.testing.assert_close(data.x_T[:, 1], torch.tensor([1., 0.]))
        examples.append(dict(row=row, data=data))
    batch, bench, y = collate(examples, ["synthetic"], device)
    model = upstream().HyperHybridGNN(23, hid=8, heads=2, n_bench=1, dropout=0.0).to(device)
    model.eval()
    together = model(batch, bench)
    separate = torch.cat([model(*collate([e], ["synthetic"], device)[:2]) for e in examples])
    torch.testing.assert_close(together, separate, atol=1e-5, rtol=1e-5)
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(together, y)
    assert torch.isfinite(loss)
    loss.backward(); opt.step()
    assert any(not torch.equal(before[k], v) for k, v in model.state_dict().items())
    with tempfile.TemporaryDirectory(dir=ROOT / "voting_adapter", prefix="test-") as tmp:
        path = Path(tmp) / "synthetic.pt"
        torch.save(model.state_dict(), path)
        clone = upstream().HyperHybridGNN(23, hid=8, heads=2, n_bench=1, dropout=0.0).to(device)
        clone.load_state_dict(torch.load(path, weights_only=True, map_location=device))
        clone.eval()
        torch.testing.assert_close(model(batch, bench), clone(batch, bench))
    print("Graph smoke passed: 2/3/5-node batches, IID invariants, forward/backward and checkpoint reload.")


if __name__ == "__main__":
    unittest.main()
