import argparse
import csv
import hashlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from conf_compose.composition.debate_records import IncompleteCell, file_digest, load_cell
from conf_compose.composition.post_debate import CompositionConfig, evaluate_cell


def parse_args():
    parser = argparse.ArgumentParser(description='Offline post-debate majority confidence composition; no inference')
    parser.add_argument('--tasks', nargs='+', help='default: all datasets present in the debate store')
    parser.add_argument('--rounds', nargs='+', type=int, default=[1])
    parser.add_argument('--estimators', nargs='+', choices=('cons', 'seq'), default=['cons', 'seq'])
    parser.add_argument('--debate-store', type=Path, default=ROOT / 'results/debate_inferences')
    parser.add_argument('--inference-store', type=Path, default=ROOT / 'results/inferences')
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'results/debate_composition')
    parser.add_argument('--match', default='', help='substring of the saved debate cell directory')
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--fit-fraction', type=float, default=0.3)
    parser.add_argument('--fit-seed', type=int, default=0)
    parser.add_argument('--tie-break', choices=('confidence', 'first'), default='confidence')
    parser.add_argument('--tie-seed', type=int, default=0)
    parser.add_argument('--context', choices=('direct', 'reasoned'), default='direct')
    parser.add_argument('--seq-score', choices=('norm_sum', 'norm_mean'), default='norm_sum')
    parser.add_argument('--no-ablations', action='store_true')
    parser.add_argument('--logistic-l2', type=float, default=1.0)
    parser.add_argument('--require-complete', action='store_true', help='fail if any selected cell is incomplete')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def discover(args):
    cells, skipped, seen = [], [], set()
    paths = sorted(args.debate_store.glob('*/*/settings.json'))
    found_tasks = set()
    for path in paths:
        settings = json.loads(path.read_text())['settings']
        if args.tasks and settings['task'] not in args.tasks or args.match not in path.parent.name:
            continue
        found_tasks.add(settings['task'])
        for round_index in dict.fromkeys(args.rounds):
            key = settings['task'], tuple(sorted(settings['group'])), round_index
            if key in seen:
                raise ValueError(f'multiple saved runs for {key}; select one with --match')
            seen.add(key)
            try:
                cell = load_cell(path.parent, round_index, args.inference_store, 'seq' in args.estimators)
            except IncompleteCell as error:
                skipped.append({'cell': str(path.parent), 'round': round_index, 'reason': str(error)})
                print(f'Skipping incomplete cell: {path.parent.name}: {error}', flush=True)
                continue
            cell.initial = {}
            cell.current = {}
            cells.append(cell)
            print(f'Checked {cell.task} r{cell.round} {"|".join(cell.models)}: {len(cell.ids)} questions', flush=True)
    if args.tasks and set(args.tasks) - found_tasks:
        raise ValueError(f'no saved cells for requested tasks: {sorted(set(args.tasks) - found_tasks)}')
    if skipped and args.require_complete:
        raise ValueError(f'{len(skipped)} selected cells are incomplete')
    if not cells:
        raise ValueError('no complete debate cells found')
    return cells, skipped


def run(args):
    config = CompositionConfig(args.samples, args.fit_fraction, args.fit_seed, args.tie_break, args.tie_seed,
                               args.context, args.seq_score, not args.no_ablations, args.logistic_l2)
    if any(round_index < 1 for round_index in args.rounds):
        raise ValueError('post-debate rounds must be positive')
    args.estimators = list(dict.fromkeys(args.estimators))
    cells, skipped = discover(args)
    print(f'{len(cells)} complete cells × {len(args.estimators)} estimators; {len(skipped)} incomplete cells', flush=True)
    if args.dry_run:
        return
    sources = {path: digest for cell in cells for path, digest in cell.sources.items()}
    code_paths = [*sorted((ROOT / 'src/conf_compose/composition').rglob('*.py')),
                  *sorted((ROOT / 'src/conf_compose/data').glob('*.py')),
                  ROOT / 'src/conf_compose/utils/metrics.py', ROOT / 'src/conf_compose/constants.json', Path(__file__)]
    code = {str(path.relative_to(ROOT)): file_digest(path) for path in code_paths}
    identity = {'config': asdict(config), 'estimators': args.estimators, 'sources': sources, 'code': code,
                'cells': [{'task': cell.task, 'models': cell.models, 'round': cell.round} for cell in cells], 'skipped': skipped,
                'runtime': {'python': sys.version, 'numpy': np.__version__, 'scipy': scipy.__version__}}
    run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    output = args.out_dir / f'run_{run_id}'
    if output.exists():
        manifest = json.loads((output / 'manifest.json').read_text())
        if manifest['identity'] != json.loads(json.dumps(identity)):
            raise ValueError(f'run identity mismatch: {output}')
        if any(not (output / name).is_file() or file_digest(output / name) != digest for name, digest in manifest['artifacts'].items()):
            raise ValueError(f'completed output was changed: {output}')
        print(f'Already complete and verified: {output}')
        return
    args.out_dir.mkdir(parents=True, exist_ok=True)
    staging = args.out_dir / f'.run_{run_id}_{os.getpid()}.partial'
    staging.mkdir()
    atomic, rows = {estimator: [] for estimator in args.estimators}, []
    for saved_cell in cells:
        cell = load_cell(saved_cell.directory, saved_cell.round, args.inference_store, 'seq' in args.estimators)
        if cell.sources != saved_cell.sources:
            raise ValueError(f'inputs changed after discovery: {cell.directory}')
        for estimator in args.estimators:
            result = evaluate_cell(cell, estimator, config)
            atomic[estimator].append({**result['atomic'], 'cell': cell.directory.name})
            rows.extend({**row, 'cell': cell.directory.name} for row in result['metrics'])
            target = staging / cell.task / cell.directory.name / f'round_{cell.round}' / estimator
            for name in ('audit', 'fits'):
                write_json(target / f'{name}.json', result[name])
            with (target / 'predictions.jsonl').open('w') as handle:
                for row in result['predictions']:
                    handle.write(json.dumps(row, allow_nan=False) + '\n')
            print(f'{cell.task} r{cell.round} {estimator} {"|".join(cell.models)}: '
                  f'fit={result["atomic"]["n_fit_scored"]}, evaluation={result["atomic"]["n_matched"]}/'
                  f'{result["atomic"]["n_evaluation"]}, vote accuracy={result["atomic"]["vote_acc"]:.3f}', flush=True)
    for path, digest in sources.items():
        if file_digest(Path(path)) != digest:
            raise ValueError(f'inference input changed during analysis: {path}; partial outputs not published')
    for estimator, entries in atomic.items():
        write_csv(staging / f'{estimator}.csv', entries)
    write_csv(staging / 'metrics.csv', rows)
    artifacts = {str(path.relative_to(staging)): file_digest(path) for path in sorted(staging.rglob('*')) if path.is_file()}
    manifest = {'identity': identity, 'run_id': run_id, 'artifacts': artifacts, 'n_cells': len(cells),
                'n_atomic_rows': sum(map(len, atomic.values())), 'n_metric_rows': len(rows),
                'inference': False, 'selection': f'post-round majority; {config.tie_break} tie-break independent of estimator',
                'fitting': 'same question-hash split as experiment02; fit independently by group, round and estimator',
                'evaluation': 'shared mask for initial solo, post-debate solo and pools within each cell and estimator',
                'unavailable_fits': 'missing scores with explicit status; never substituted',
                'reference': 'best initial solo by fitting accuracy, fixed across metrics',
                'skipped': skipped}
    write_json(staging / 'manifest.json', manifest)
    staging.rename(output)
    print(f'Completed: {output}')


if __name__ == '__main__':
    run(parse_args())
