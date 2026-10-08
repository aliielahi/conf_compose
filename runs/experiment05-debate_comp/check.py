import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from conf_compose.composition.debate_records import file_digest
from conf_compose.composition.pooling import FittedBLP, FittedPool, pool_methods
from conf_compose.composition.post_debate import METRICS, metric_values


def equal(actual, expected, label):
    if actual is None or expected is None:
        if actual is not expected:
            raise ValueError(f'{label}: {actual} != {expected}')
    elif not math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError(f'{label}: {actual} != {expected}')


def check(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    for name, digest in manifest['artifacts'].items():
        if file_digest(directory / name) != digest:
            raise ValueError(f'changed output: {name}')
    for name, digest in manifest['identity']['sources'].items():
        if file_digest(Path(name)) != digest:
            raise ValueError(f'changed input: {name}')
    with (directory / 'metrics.csv').open() as handle:
        metrics = list(csv.DictReader(handle))
    targets, n_cells = {}, 0
    for path in sorted(directory.glob('*/*/round_*/*/audit.json')):
        audit = json.loads(path.read_text())
        fits = json.loads(path.with_name('fits.json').read_text())
        fits = {name: (FittedBLP if name in ('blp', 'blp_equal') else FittedPool)(**values) for name, values in fits.items()}
        predictions = [json.loads(line) for line in path.with_name('predictions.jsonl').read_text().splitlines()]
        by_id = {row['id']: row for row in predictions}
        fitting, evaluation = set(audit['fit_ids']), set(audit['eval_ids'])
        if fitting & evaluation or fitting | evaluation != set(by_id) or len(by_id) != len(predictions):
            raise ValueError(f'question partition mismatch: {path}')
        for row in predictions:
            question = row['id']
            if row['matched'] != (question in audit['matched_ids']):
                raise ValueError(f'matched mask mismatch: {path}, {question}')
            key = audit['task'], audit['models'], audit['round'], question
            target = row['post_debate']['target']
            if key in targets and targets[key] != target:
                raise ValueError(f'estimators changed the selected answer: {key}')
            targets[key] = target
            if row['scored']:
                shared = row['post_debate']['shared']
                expected = {**pool_methods(shared), **{name: fit.predict(shared) for name, fit in fits.items()}}
                for name, prediction in expected.items():
                    equal(row['pooled'][name]['confidence'], prediction.score, f'{key}/{name}')
        selected = [row for row in metrics if row['task'] == audit['task'] and row['models'] == audit['models']
                    and int(row['round']) == audit['round'] and row['estimator'] == audit['estimator']]
        for entry in selected:
            method = entry['method']
            if method == 'best_solo':
                method = f'initial:{audit["reference_model"]}'
            probabilities, labels, ranks = [], [], []
            for question in audit['matched_ids']:
                row = by_id[question]
                if ':' in method:
                    stage, model = method.split(':')
                    index = audit['models'].split('|').index(model)
                    value = row[stage]['own'][index]
                    correct = row['initial_correct' if stage == 'initial' else 'post_correct'][index]
                    rank = value
                else:
                    value = row['pooled'][method]['confidence']
                    rank = row['pooled'][method]['ranking']
                    correct = row['correct']
                if value is not None:
                    probabilities.append(value)
                    labels.append(correct)
                    ranks.append(rank)
            values = metric_values(probabilities, labels, ranks)
            for metric in METRICS:
                equal(float(entry[metric]) if entry[metric] else None, values[metric], f'{path}/{method}/{metric}')
            equal(entry['coverage'], len(probabilities) / len(evaluation), f'{path}/{method}/coverage')
        n_cells += 1
    if n_cells != manifest['n_atomic_rows'] or len(metrics) != manifest['n_metric_rows']:
        raise ValueError('manifest row counts do not match outputs')
    print(f'Verified {n_cells} group/estimator cases and {len(metrics)} metric rows: input/output hashes, splits, targets, pooling replay and metrics.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Verify saved debate composition without refitting or inference')
    parser.add_argument('--run', type=Path, required=True)
    check(parser.parse_args().run)
