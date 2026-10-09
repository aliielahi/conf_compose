import unittest
import tempfile
from pathlib import Path
from statistics import stdev

from tables import FULL_METRICS, aggregate, display, render_table, write_paper_table
from conf_compose.utils.calibration import fit_temperature, temperature_scale
from conf_compose.utils.metrics import nll


class DeltaSpreadTest(unittest.TestCase):
    def test_standard_deviation_uses_group_deltas(self):
        cells = [{"estimator": "cons", "task": "csqa", "n_models": size} for size in (2, 2, 3, 3)]
        deltas = [-0.3, -0.1, 0.1, 0.5]
        rows = [{**cell, "method": "mean", "delta_ece": value} for cell, value in zip(cells, deltas)]
        values = aggregate(rows, cells, "cons", None, ["mean"], ["csqa"], True, ("ece",), spread=True)
        average, n, total, sd = values["mean", "csqa", "ece"]
        self.assertAlmostEqual(average, 0.05)
        self.assertEqual((n, total), (4, 4))
        self.assertAlmostEqual(sd, stdev(deltas))
        restricted = aggregate(rows, cells, "cons", 2, ["mean"], ["csqa"], True, ("ece",), spread=True)
        self.assertAlmostEqual(restricted["mean", "csqa", "ece"][3], stdev(deltas[:2]))

    def test_units_and_latex(self):
        cell = {"estimator": "cons", "task": "csqa", "n_models": 2}
        rows = [{**cell, "method": "mean", **{f"delta_{metric}": value for metric in FULL_METRICS}}
                for value in (0.1, 0.2)]
        values = aggregate(rows, [cell, cell], "cons", None, ["mean"], ["csqa"], True, FULL_METRICS, spread=True)
        text, latex = render_table(values, {"mean": "Mean"}, ["csqa"], "test", True, FULL_METRICS)
        self.assertIn('+15.00 ± 7.07', text)
        self.assertIn('+0.150 ± 0.071', text)
        self.assertIn(r'{\scriptsize $\pm$ 7.07}', latex)
        self.assertIn('not a confidence interval', latex)

    def test_paper_table_uses_group_deltas_for_every_metric(self):
        fields = ("t_ece", "t_brier", "auarc", "answer_matched_auarc")
        cells = [{"estimator": "cons", "task": "csqa", "n_models": 2} for _ in range(2)]
        rows = [{**cell, "method": "mean", **dict.fromkeys(fields, absolute),
                 **{f"delta_{field}": delta for field in fields}}
                for cell, absolute, delta in zip(cells, (0.2, 0.8), (-0.1, 0.3))]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "table.tex"
            write_paper_table(path, rows, cells, "cons", {"mean": "Average"}, ["csqa"])
            text = path.read_text()
        self.assertEqual(text.count(r"+10.00 {\scriptsize $\pm$ 28.28}"), 4)
        self.assertNotIn("Absolute metrics", text)
        self.assertIn(r"$\Delta$t-ECE", text)

    def test_missing_and_single_group(self):
        self.assertEqual(display(None, 0, 15, None, delta=True), '--')
        self.assertEqual(display(0.1, 1, 15, None, delta=True), '+10.00')
        self.assertEqual(display(0, 15, 15, 0, delta=True), '+0.00 ± 0.00')

    def test_absolute_unchanged(self):
        self.assertEqual(display(0.1, 15, 15, 0.2, delta=False), '10.00')
        self.assertEqual(display(0.1, 15, 15, 0.2, delta=False, metric='nll'), '0.100')

    def test_temperature_changes_probabilities_without_changing_order(self):
        scores = [0.1, 0.3, 0.6, 0.8, 0.9]
        labels = [0, 1, 0, 1, 1]
        temperature = fit_temperature(scores, labels)
        scaled = temperature_scale(scores, temperature)
        self.assertLessEqual(nll(scaled, labels), nll(scores, labels) + 1e-12)
        self.assertEqual(list(sorted(range(len(scores)), key=lambda i: scores[i])),
                         list(sorted(range(len(scores)), key=lambda i: scaled[i])))


if __name__ == '__main__':
    unittest.main()
