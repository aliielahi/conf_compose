import json
import tempfile
import unittest
from pathlib import Path

from run import reference_stream, summary, paper_table


class DebateReportTests(unittest.TestCase):
    def test_reference_scores_the_selected_answer(self):
        with tempfile.TemporaryDirectory() as directory:
            predictions = Path(directory) / 'predictions.jsonl'
            records = [
                {'id': 'fit', 'correct': True, 'matched': False,
                 'post_debate': {'target': 'Yes', 'shared': [0.8, 0.7]},
                 'post_answers': ['Yes', 'Yes']},
                {'id': 'right', 'correct': True, 'matched': True,
                 'post_debate': {'target': 'Yes', 'shared': [0.9, 0.6]},
                 'post_answers': ['Yes', 'Yes']},
                {'id': 'wrong', 'correct': False, 'matched': True,
                 'post_debate': {'target': 'No', 'shared': [0.4, 0.3]},
                 'post_answers': ['No', 'Yes']},
            ]
            predictions.write_text(''.join(json.dumps(record) + '\n' for record in records))
            audit = {'task': 'boolq', 'models': 'a|b', 'round': 1, 'estimator': 'cons',
                     'reference_model': 'a', 'n_models': 2, 'n_evaluation': 2, 'n_matched': 2,
                     'fit_ids': ['fit'], 'fit_scored_ids': ['fit'], 'eval_ids': ['right', 'wrong'],
                     'matched_ids': ['right', 'wrong'], 'pool_accuracy_all': 0.5}
            row = reference_stream(audit, predictions)
            self.assertEqual(row['accuracy_all'], 0.5)
            self.assertAlmostEqual(row['auroc'], 1)

    def test_delta_and_spread_are_across_matched_groups(self):
        rows = [{'method': 'mean', 'task': 'boolq', 'models': model, 'round': 1,
                 'n_models': 2, 'ece': score, 'delta_ece': delta}
                for model, score, delta in [('a|b', 0.2, -0.1), ('c|d', 0.4, 0.1)]]
        audits = [{'task': 'boolq', 'n_models': 2} for _ in rows]
        value, spread, count, expected = summary(rows, audits, 'mean', 'boolq', 2, 'ece', 'target')
        self.assertAlmostEqual(value, 0)
        self.assertAlmostEqual(spread, 2 ** 0.5 / 10)
        self.assertEqual((count, expected), (2, 2))

    def test_paper_table_uses_group_deltas_for_every_metric(self):
        fields = ("t_ece", "t_brier", "auarc", "answer_matched_auarc")
        rows = [{"method": "mean", "task": "boolq", "models": model,
                 "round": 1, "n_models": 2, **dict.fromkeys(fields, absolute),
                 **{"delta_" + field: delta for field in fields}}
                for model, absolute, delta in (("a|b", 0.2, -0.1), ("c|d", 0.8, 0.3))]
        audits = [{"task": "boolq", "n_models": 2} for _ in rows]
        text = paper_table(rows, audits, ["boolq"])
        self.assertEqual(text.count(r"+10.00 {\scriptsize $\pm$ 28.28}"), 4)
        self.assertNotIn("Absolute metrics", text)
        self.assertIn(r"$\Delta$t-Brier", text)

    def test_target_comparison_uses_the_same_group(self):
        rows = [{'method': method, 'task': 'boolq', 'models': model, 'round': 1,
                 'n_models': 2, 'ece': score, 'delta_ece': score - target}
                 for model, target, pooled in [('a|b', 0.3, 0.2), ('c|d', 0.5, 0.3)]
                 for method, score in [('reference_stream_target', target), ('mean', pooled)]]
        audits = [{'task': 'boolq', 'n_models': 2} for _ in range(2)]
        value, spread, count, expected = summary(rows, audits, 'mean', 'boolq', 2, 'ece', 'target')
        self.assertAlmostEqual(value, -0.15)
        self.assertAlmostEqual(spread, 2 ** 0.5 / 20)
        self.assertEqual((count, expected), (2, 2))


if __name__ == '__main__':
    unittest.main()
