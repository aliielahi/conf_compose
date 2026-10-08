import copy
import json
import tempfile
import unittest
from pathlib import Path

from cagecal import append_results, ids_digest
from tables import FULL_METRICS, write_csv
from conf_compose.utils import metrics


class CagecalTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        (self.directory / 'paper_tables').mkdir()
        self.cell = dict(task='csqa', models='a|b', estimator='cons', n_models=2, n_matched=2,
                         matched_ids=['q1', 'q2'], fit_ids_hash=ids_digest(['t1', 'v1']),
                         eval_ids_hash=ids_digest(['q1', 'q2']), tie_break='confidence', tie_seed=0,
                         selected_targets={'q1': 'A', 'q2': 'B'}, vote_accuracy=0.5)
        self.rows = [dict(self.cell, method='best_solo', **{f'reference_{metric}': 0.25 for metric in FULL_METRICS})]
        self.predictions = [dict(task='csqa', models='a|b', id=question, target=target, correct=correct, raw=p, betasb=p)
                            for question, target, correct, p in [('q1', 'A', 1, 0.8), ('q2', 'B', 0, 0.2)]]
        self.manifest = dict(identity='test', config={'seeds': 1},
                             splits={'csqa': {'train': ['t1'], 'validation': ['v1'], 'evaluation': ['q1', 'q2']}})
        self.save('manifest.json', self.manifest)
        self.save('calibration.json', {})
        self.save('paper_tables/audit.json', [self.cell])
        self.save_predictions()
        values = {metric: getattr(metrics, metric)([0.8, 0.2], [1, 0]) for metric in FULL_METRICS if metric != 'accuracy'}
        write_csv(self.directory / 'paper_tables/cagecal_metrics.csv',
                  [dict(task='csqa', models='a|b', estimator='cons', method=method, accuracy=0.5, **values)
                   for method in ('cagecal_iid', 'cagecal_iid_betasb')])

    def save(self, filename, value):
        (self.directory / filename).write_text(json.dumps(value))

    def save_predictions(self):
        (self.directory / 'predictions.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in self.predictions))

    def test_deltas_and_idempotence(self):
        rows, labels, provenance = append_results(self.rows, [self.cell], self.directory)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[-1]['accuracy'], 0.5)
        self.assertAlmostEqual(rows[-1]['delta_brier'], 0.04 - 0.25)
        self.assertEqual(provenance['n_cells'], 1)
        again, _, _ = append_results(rows, [self.cell], self.directory)
        self.assertEqual(rows, again)
        raw, _, _ = append_results(again, [self.cell], self.directory, 'raw')
        self.assertEqual(len(raw), 2)
        self.assertEqual(raw[-1]['method'], 'cagecal_iid')

    def test_target_mismatch(self):
        cell = copy.deepcopy(self.cell)
        cell['selected_targets']['q1'] = 'B'
        with self.assertRaisesRegex(ValueError, 'target mismatch'):
            append_results(self.rows, [cell], self.directory)

    def test_split_overlap(self):
        self.manifest['splits']['csqa']['train'].append('q1')
        self.save('manifest.json', self.manifest)
        with self.assertRaisesRegex(ValueError, 'split overlap'):
            append_results(self.rows, [self.cell], self.directory)

    def test_mask_mismatch(self):
        cell = dict(self.cell, matched_ids=['q1'])
        with self.assertRaisesRegex(ValueError, 'matched_ids mismatch'):
            append_results(self.rows, [cell], self.directory)

    def test_missing_and_duplicate_predictions(self):
        self.predictions.pop()
        self.save_predictions()
        with self.assertRaises(KeyError):
            append_results(self.rows, [self.cell], self.directory)
        self.predictions.append(self.predictions[0])
        self.save_predictions()
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            append_results(self.rows, [self.cell], self.directory)

    def test_modified_scores(self):
        self.predictions[0]['betasb'] = 0.3
        self.save_predictions()
        with self.assertRaisesRegex(ValueError, 'does not match predictions'):
            append_results(self.rows, [self.cell], self.directory)


if __name__ == '__main__':
    unittest.main()
