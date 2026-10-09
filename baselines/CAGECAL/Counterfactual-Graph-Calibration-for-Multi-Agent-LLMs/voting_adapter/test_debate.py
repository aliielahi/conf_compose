"""CPU regression tests for paired debate inputs and validation-aware reporting."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from .data import ROOT, DEFAULT_PROJECT, file_hash
from .debate_data import load_debate_bundle


class Task:
    name = 'csqa'
    def equivalent(self, a, b):
        return a == b


class DebateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / 'voting_adapter', prefix='test-')
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name)
        self.table = self.project / 'paper_results/results/debate_protocol/run_test/cons'
        self.run = self.project / 'results/debate_composition/run_test'
        self.cell = self.project / 'results/debate_inferences/csqa/a+b'
        self.group = self.run / 'csqa/a+b/round_1/cons'
        self.ids = [str(i) for i in range(13)]
        self.sources = []
        self.initial = []
        self.current = []
        def write(path, value):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))
        self.write = write
        def jsonl(path, rows):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        self.jsonl = jsonl
        round0 = {}
        for i, alias in enumerate(('a', 'b')):
            before, after = [], []
            for q in self.ids:
                initial = 'A' if i == 0 else 'B'
                final = 'A' if int(q) % 2 else 'B'
                common = dict(id=q, question='question ' + q, gold='A', options=['A', 'B'], choices={'A': 'one', 'B': 'two'}, token_logprobs={'response': [-.5, -.7]})
                before.append(dict(common, prediction=initial, correct=initial == 'A', sampled_answers={'consistency_t0.7': [initial]*5}))
                after.append(dict(common, prediction=final, correct=final == 'A', sampled_answers={'consistency_t0.7': [final]*5}, round=1, model=alias, group=['a', 'b'], peer_order=['b' if i == 0 else 'a'], error=None))
            p0 = self.project / f'results/inferences/csqa/{alias}/test.jsonl'
            p1 = self.cell / f'round_1/{alias}.jsonl'
            jsonl(p0, before); jsonl(p1, after)
            self.initial.append(p0); self.current.append(p1); self.sources.extend([p0, p1])
            round0[alias] = dict(path=str(p0), sha256=file_hash(p0))
        write(self.cell / 'settings.json', dict(settings=dict(task='csqa', group=['a', 'b']), round0=round0))
        self.sources.append(self.cell / 'settings.json')
        audit = dict(task='csqa', models='a|b', round=1, n_models=2, matched_ids=self.ids[10:], fit_ids=self.ids[:10], eval_ids=self.ids[10:], config=dict(samples=5, tie_break='first', tie_seed=0))
        write(self.group / 'audit.json', audit)
        predictions = [dict(id=q, split='fitting' if int(q)<10 else 'evaluation', correct=bool(int(q)%2),
                            initial=dict(target='A', selection_reason='first'),
                            post_debate=dict(target='A' if int(q)%2 else 'B', selection_reason='count')) for q in self.ids]
        jsonl(self.group / 'predictions.jsonl', predictions)
        write(self.table / 'audit.json', [audit])
        (self.table / 'atomic.csv').write_text('task,models,cell,estimator\ncsqa,a|b,a+b,cons\n')
        self.run.mkdir(parents=True, exist_ok=True)
        (self.run / 'metrics.csv').write_text('test\n')
        self.refresh()

    def refresh(self):
        artifacts = {str(p.relative_to(self.run)): file_hash(p) for p in self.group.iterdir()}
        self.write(self.run / 'manifest.json', dict(identity=dict(sources={str(p): file_hash(p) for p in self.sources}), artifacts=artifacts))
        self.write(self.table / 'manifest.json', dict(source_run=str(self.run), source_manifest_sha256=file_hash(self.run/'manifest.json'), source_metrics_sha256=file_hash(self.run/'metrics.csv')))

    def load(self):
        with patch('voting_adapter.debate_data.project_api', return_value=lambda name: Task()):
            return load_debate_bundle(self.project, table_dir=self.table)

    def test_paired_records_splits_and_communication(self):
        bundle = self.load()
        self.assertEqual(len(bundle['rows']), 13)
        row = bundle['rows'][0]
        self.assertEqual(row['initial_answers'], ['A', 'B'])
        self.assertEqual(row['answers'], ['B', 'B'])
        self.assertEqual(row['communication'], [[0, 1], [1, 0]])
        splits = bundle['splits']['csqa']
        self.assertFalse(set(splits['train']) & set(splits['validation']))
        self.assertFalse(set(splits['evaluation']) & (set(splits['train']) | set(splits['validation'])))
        self.assertAlmostEqual(bundle['cells'][0]['vote_accuracy'], 1/3)

    def test_changed_inference_hash_is_rejected(self):
        with self.current[0].open('a') as f:
            f.write('\n')
        with self.assertRaisesRegex(ValueError, 'Source hash differs'):
            self.load()

    def test_missing_matched_features_are_not_silently_dropped(self):
        rows = [json.loads(line) for line in self.current[0].read_text().splitlines()]
        rows[-1]['token_logprobs'] = {}
        self.jsonl(self.current[0], rows); self.refresh()
        with self.assertRaisesRegex(ValueError, 'matched examples lack'):
            self.load()

    def test_changed_peer_visibility_is_rejected(self):
        rows = [json.loads(line) for line in self.current[0].read_text().splitlines()]
        rows[0]['peer_order'] = []
        self.jsonl(self.current[0], rows); self.refresh()
        with self.assertRaisesRegex(ValueError, 'full peer exchange'):
            self.load()

    def test_validation_temperature_and_no_old_baseline_rows(self):
        from .report import make_reports
        with tempfile.TemporaryDirectory(dir=ROOT / 'voting_adapter', prefix='test-') as tmp:
            tmp = Path(tmp)
            base = dict(task='csqa', models='a|b', estimator='cons', n_models=2, n_matched=2, coverage=1., method='mean', output_temperature=9.)
            for metric in ('accuracy', 'ece', 'auarc', 'auroc', 'brier', 'nll', 't_brier', 't_ece'):
                base.update({metric: .4, 'reference_'+metric: .3, 'delta_'+metric: .1})
            source = tmp/'atomic.csv'
            with source.open('w') as f:
                writer = csv.DictWriter(f, fieldnames=list(base)); writer.writeheader()
                writer.writerows([base, dict(base, method='cagecal_iid_betasb')])
            rows = [dict(task='csqa', models='a|b', id=str(i), target='A', correct=i%2, split='validation' if i<4 else 'evaluation') for i in range(6)]
            cell = dict(task='csqa', models='a|b', estimator='cons', n_models=2, matched_ids=['4','5'], coverage=1., vote_accuracy=.5)
            bundle = dict(project=DEFAULT_PROJECT, protocol='debate', rows=rows, cells=[cell], atomic_path=source)
            pred = tmp/'predictions.jsonl'
            self.jsonl(pred, [dict(r, raw=.4 if not r['correct'] else .6, betasb=.3 if not r['correct'] else .7) for r in rows[4:]])
            self.jsonl(tmp/'validation_predictions.jsonl', [dict(r, raw=.4 if not r['correct'] else .6) for r in rows[:4]])
            report = make_reports(bundle, pred, tmp/'tables')
            self.assertLess(report[0]['t_brier'], report[0]['brier'])
            self.assertIsNone(report[1]['t_brier'])
            with (tmp/'tables/atomic.csv').open() as handle:
                out = list(csv.DictReader(handle))
            self.assertEqual(len(out), 3)
            self.assertEqual({r['method'] for r in out}, {'mean','cagecal_debate','cagecal_debate_betasb'})
            self.assertNotIn('Voting protocol', (tmp/'tables/cons/all_absolute.txt').read_text())
            self.assertNotEqual(float(out[1]['output_temperature']), 9.)
            # Evaluation records must never be accepted as calibration inputs.
            self.jsonl(tmp/'validation_predictions.jsonl', [dict(rows[4], raw=.4)])
            with self.assertRaisesRegex(ValueError, 'validation IDs'):
                make_reports(bundle, pred, tmp/'bad')


if __name__ == '__main__':
    unittest.main()
