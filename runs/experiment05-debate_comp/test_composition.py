import copy
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from conf_compose.composition.debate_records import IncompleteCell, file_digest, load_cell
from conf_compose.composition.post_debate import CompositionConfig, evaluate_cell, score_question, sequence_score, split_ids
from conf_compose.data import get_task


class DebateCompositionTest(unittest.TestCase):
    def setUp(self):
        temporary_root = ROOT / 'results/debate_composition'
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=temporary_root)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.directory = self.root / 'debate/gpqa/a+b'
        self.store = self.root / 'inferences'
        self.models = ('a', 'b')
        self.config = CompositionConfig()
        self.task = get_task('gpqa')
        self.settings = {'settings': {'task': 'gpqa', 'group': list(self.models), 'limit': None}, 'round0': {}}
        for agent, model in enumerate(self.models):
            initial, current, scores = [], [], []
            for number in range(30):
                question = f'gpqa-train-{number}'
                answer = ('A', 'B')[(number + agent) % 2]
                gold = ('A', 'B', 'C')[number % 3]
                row = {'id': question, 'question': f'Question {number}', 'gold': gold, 'prediction': answer,
                       'correct': answer == gold, 'options': list('ABCD'), 'choices': {label: label for label in 'ABCD'},
                       'sampled_answers': {'consistency_t0.7': [answer] * (agent + 2) + ['C'] * (3 - agent)},
                       'candidate_scores': self.candidates('A'), 'error': None}
                initial.append(row)
                current.append({**row, 'round': 1, 'model': model, 'agent': agent, 'group': list(self.models),
                                'messages': [{'role': 'user', 'content': 'Peer response'}]})
                scores.extend([{'id': question, 'round': r, 'candidate_scores': self.candidates(label)}
                               for r, label in [(0, 'A'), (1, 'B')]])
            path = self.store / 'gpqa' / f'vllm__{model}--source' / 'test.jsonl'
            self.write_rows(path, initial)
            self.settings['round0'][model] = {'path': f'/workspace/results/inferences/gpqa/{path.parent.name}/test.jsonl',
                                              'sha256': file_digest(path)}
            self.write_rows(self.directory / 'round_1' / f'{model}.jsonl', current)
            self.write_rows(self.directory / 'candidates_through_round_1' / f'{model}.jsonl', scores)
        self.save_settings()

    def candidates(self, favored):
        return {'candidates': [{'answer': label, 'option': label, 'direct': {'logprobs': [math.log(0.7 if label == favored else 0.1)]},
                                'reasoned': {'logprobs': [math.log(0.4 if label == favored else 0.2)]}} for label in 'ABCD']}

    def write_rows(self, path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))

    def save_settings(self):
        (self.directory / 'settings.json').write_text(json.dumps(self.settings))

    def load(self):
        return load_cell(self.directory, 1, self.store)

    def test_round_specific_scores_and_relocated_source(self):
        cell = self.load()
        row = cell.current['a'][cell.ids[0]]
        self.assertAlmostEqual(sequence_score(self.task, row, 'B', self.config), 0.7)
        self.assertAlmostEqual(sequence_score(self.task, cell.initial['a'][cell.ids[0]], 'B', self.config), 0.1)

    def test_source_hash_and_duplicate_ids(self):
        path = self.store / 'gpqa/vllm__a--source/test.jsonl'
        path.write_text(path.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.load()
        self.settings['round0']['a']['sha256'] = file_digest(path)
        self.save_settings()
        path = self.directory / 'round_1/a.jsonl'
        path.write_text(path.read_text() + path.read_text().splitlines()[0] + '\n')
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.load()

    def test_missing_candidate_file_and_partial_questions(self):
        path = self.directory / 'candidates_through_round_1/a.jsonl'
        path.unlink()
        with self.assertRaises(IncompleteCell):
            self.load()
        path = self.directory / 'round_1/a.jsonl'
        path.write_text('\n'.join(path.read_text().splitlines()[1:]) + '\n')
        with self.assertRaises(IncompleteCell):
            load_cell(self.directory, 1, self.store, False)

    def test_wrong_round_and_candidate_mapping(self):
        path = self.directory / 'round_1/a.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]['round'] = 0
        self.write_rows(path, rows)
        with self.assertRaisesRegex(ValueError, 'metadata'):
            self.load()
        rows[0]['round'] = 1
        self.write_rows(path, rows)
        path = self.directory / 'candidates_through_round_1/a.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[1]['candidate_scores']['candidates'][0]['option'] = 'wrong answer text'
        self.write_rows(path, rows)
        with self.assertRaisesRegex(ValueError, 'mapping'):
            self.load()

    def test_shared_target_and_zero_support(self):
        cell = self.load()
        question = cell.ids[0]
        for model, answer in [('a', 'A'), ('b', 'B')]:
            cell.current[model][question]['prediction'] = answer
            cell.current[model][question]['sampled_answers']['consistency_t0.7'] = [answer] * (5 if model == 'a' else 2) + ([] if model == 'a' else ['C'] * 3)
        cons = score_question(self.task, cell.models, cell.current, question, 1, 'cons', self.config)
        seq = score_question(self.task, cell.models, cell.current, question, 1, 'seq', self.config)
        self.assertEqual(cons['target'], 'A')
        self.assertEqual(seq['target'], 'A')
        self.assertAlmostEqual(cons['shared'][1], 0.5 / 6)
        cell.current['b'][question]['sampled_answers']['consistency_t0.7'] = [None] * 5
        missing = score_question(self.task, cell.models, cell.current, question, 1, 'cons', self.config)
        self.assertIsNone(missing['shared'][1])
        self.assertEqual(missing['selection_reason'], 'missing_consistency_seeded')

    def test_incomplete_label_scores_fail(self):
        cell = self.load()
        row = copy.deepcopy(cell.current['a'][cell.ids[0]])
        row['candidate_scores']['candidates'].pop()
        with self.assertRaisesRegex(ValueError, 'closed-label'):
            sequence_score(self.task, row, 'A', self.config)

    def test_heldout_labels_never_enter_fit(self):
        cell = self.load()
        fitting, evaluation = split_ids(cell.task, cell.ids, self.config)
        with patch('conf_compose.composition.post_debate.fit_methods', return_value={}) as fit:
            first = evaluate_cell(cell, 'cons', self.config)
            first_scores, first_labels = fit.call_args.args
        for q in evaluation:
            for records in (cell.initial, cell.current):
                for model in cell.models:
                    records[model][q]['gold'] = 'D'
                    records[model][q]['correct'] = False
        with patch('conf_compose.composition.post_debate.fit_methods', return_value={}) as fit:
            second = evaluate_cell(cell, 'cons', self.config)
            self.assertEqual(first_labels, fit.call_args.args[1])
            self.assertTrue((first_scores == fit.call_args.args[0]).all())
        self.assertEqual(first['audit']['reference_model'], second['audit']['reference_model'])
        self.assertFalse(set(fitting) & set(evaluation))
        pool = [r for r in first['metrics'] if r['method'] in ('mean', 'logodds_sum', 'logodds_mean')]
        self.assertEqual(len({r['accuracy'] for r in pool}), 1)
        self.assertEqual(pool[1]['auarc'], pool[2]['auarc'])

    def test_failed_turn_kept_in_accuracy_and_excluded_from_confidence(self):
        cell = self.load()
        _, evaluation = split_ids(cell.task, cell.ids, self.config)
        question = evaluation[0]
        cell.current['a'][question].update(prediction=None, correct=False, error='context_overflow')
        with patch('conf_compose.composition.post_debate.fit_methods', return_value={}):
            result = evaluate_cell(cell, 'cons', self.config)
        self.assertEqual(result['atomic']['n_evaluation'], len(evaluation))
        self.assertIn(question, result['audit']['excluded'])
        self.assertNotIn(question, result['audit']['matched_ids'])
        self.assertEqual(len(result['predictions']), len(cell.ids))

    def test_voting_runner_parity(self):
        sys.path.insert(0, str(ROOT / 'runs/experiment02-voting_composition'))
        import atomic
        cell = self.load()
        records = {f'vllm/{m}': cell.current[m] for m in cell.models}
        items = atomic.group_items(records, list(records))
        fitting, evaluation = atomic.split_items(cell.task, items, 0.3, 0)
        self.assertEqual(split_ids(cell.task, cell.ids, self.config),
                         ([i.example_id for i in fitting], [i.example_id for i in evaluation]))
        for estimator in ('cons', 'seq'):
            args = SimpleNamespace(estimator=estimator, context='direct', seq_score='norm_sum', samples=5,
                                   tie_break='confidence', tie_seed=0)
            for item in items:
                target, own, shared = atomic.confidence_scores(self.task, item, args, records)
                actual = score_question(self.task, cell.models, cell.current, item.example_id, 1, estimator, self.config)
                self.assertEqual(target, actual['target'])
                self.assertEqual(own, actual['own'])
                self.assertEqual(shared, actual['shared'])

    def test_degenerate_fit_is_unavailable(self):
        cell = self.load()
        for question in cell.ids:
            target = score_question(self.task, cell.models, cell.current, question, 1, 'cons', self.config)['target']
            for records in (cell.initial, cell.current):
                for model in cell.models:
                    records[model][question]['gold'] = target
                    records[model][question]['correct'] = records[model][question]['prediction'] == target
        result = evaluate_cell(cell, 'cons', self.config)
        for row in result['metrics']:
            if row['method'] in result['fits']:
                self.assertEqual(row['coverage'], 0)
                self.assertIsNone(row['ece'])
                self.assertEqual(row['status'], 'insufficient_classes')
        self.assertEqual(result['atomic']['vote_acc'], 1.0)

    def test_open_answer_equivalence_and_conditional_scoring(self):
        task = get_task('gsm8k')
        records = {m: {'q': {'id': 'q', 'prediction': answer,
                            'sampled_answers': {'consistency_t0.7': ['12.0'] * 5},
                            'candidate_scores': {'candidates': [{'answer': '12', 'direct': {'logprobs': [-1.0]}},
                                                                {'answer': '7', 'direct': {'logprobs': [-2.0]}}]}}}
                   for m, answer in [('a', '12'), ('b', '12.0')]}
        result = score_question(task, ('a', 'b'), records, 'q', 1, 'cons', self.config)
        self.assertEqual(result['shared'], [5.5 / 6, 5.5 / 6])
        probability = sequence_score(task, records['b']['q'], '12.0', self.config)
        self.assertAlmostEqual(probability, 1 / (1 + math.exp(-1)))


if __name__ == '__main__':
    unittest.main()
