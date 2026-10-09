"""Read paired round-0/debate records and the existing composition audit.

No model imports or writes. The paper's IDs/targets are authoritative and are
checked against the raw records; neither missing features nor ties change them.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from .data import (DEFAULT_PROJECT, file_hash, internal_splits, portable_source,
                   project_api, read_records, select_vote)


def load_debate_bundle(project=DEFAULT_PROJECT, tasks=None, panel=None,
                       val_fraction=0.2, seed=0, table_dir=None):
    project = Path(project).resolve()
    get_task = project_api(project)
    if table_dir is None:
        candidates = sorted((project / 'paper_results/results/debate_protocol').glob('*/cons/manifest.json'))
        if len(candidates) != 1:
            raise ValueError('Specify --debate-table: expected exactly one debate paper report')
        table_dir = candidates[0].parent
    table_dir = Path(table_dir).resolve()
    report = json.loads((table_dir / 'manifest.json').read_text())
    run = portable_source(project, report['source_run'])
    provenance, cache = {}, {}

    def checked(path, expected=None):
        path = Path(path).resolve()
        key = str(path.relative_to(project))
        actual = provenance.get(key)
        if actual is None:
            actual = file_hash(path)
            provenance[key] = actual
        if expected is not None and actual != expected:
            raise ValueError(f'Source hash differs: {path}')
        return path

    checked(run / 'manifest.json', report['source_manifest_sha256'])
    manifest = json.loads((run / 'manifest.json').read_text())
    checked(run / 'metrics.csv', report['source_metrics_sha256'])
    for name in ('manifest.json', 'atomic.csv', 'audit.json'):
        checked(table_dir / name)
    source_hashes = {portable_source(project, key): value for key, value in manifest['identity']['sources'].items()}
    artifacts = manifest['artifacts']

    def artifact(path):
        relative = str(path.relative_to(run))
        if relative not in artifacts:
            raise ValueError(f'Unmanifested composition artifact: {path}')
        return checked(path, artifacts[relative])

    def records(path):
        path = Path(path).resolve()
        if path not in source_hashes:
            raise ValueError(f'Inference absent from composition manifest: {path}')
        checked(path, source_hashes[path])
        if path not in cache:
            print(f'Reading {path.relative_to(project)}', flush=True)
            cache[path] = read_records(path)
        return cache[path]

    with (table_dir / 'atomic.csv').open() as f:
        atomic = list(csv.DictReader(f))
    audits = json.loads((table_dir / 'audit.json').read_text())
    selected = [c for c in audits if (not tasks or c['task'] in tasks)
                and (not panel or c['models'] == '|'.join(panel))]
    if not selected or (tasks and set(tasks) != {c['task'] for c in selected}):
        raise ValueError('Requested tasks/panel missing from debate paper table')
    if {int(c['round']) for c in selected} != {1}:
        raise ValueError('This adapter currently supports paired round-0 / round-1 debate only')
    rows, cells, groups, splits, seen = [], [], [], {}, set()
    for paper_cell in selected:
        task_name, models = paper_cell['task'], paper_cell['models']
        if (task_name, models) in seen:
            raise ValueError('Multiple rounds/estimators in one debate report; select one table')
        seen.add((task_name, models))
        task = get_task(task_name)
        aliases = models.split('|')
        model_ids = ['vllm/' + alias for alias in aliases]
        sample = next(r for r in atomic if r['task'] == task_name and r['models'] == models)
        directory = project / 'results/debate_inferences' / task_name / sample['cell']
        settings_path = directory / 'settings.json'
        checked(settings_path, source_hashes[settings_path])
        settings = json.loads(settings_path.read_text())
        if settings['settings']['group'] != aliases or settings['settings']['task'] != task_name:
            raise ValueError('Debate model order/task differs from paper')
        root = run / task_name / sample['cell'] / 'round_1' / sample['estimator']
        audit = json.loads(artifact(root / 'audit.json').read_text())
        saved = {}
        with artifact(root / 'predictions.jsonl').open() as f:
            for line in f:
                record = json.loads(line)
                if record['id'] in saved:
                    raise ValueError('Duplicate saved composition prediction')
                saved[record['id']] = record
        fit, evaluation = set(audit['fit_ids']), set(audit['eval_ids'])
        if fit & evaluation or set(saved) != fit | evaluation:
            raise ValueError('Overlapping or incomplete composition splits')
        train, validation = internal_splits(task_name, fit, seed, val_fraction)
        split = dict(train=sorted(train), validation=sorted(validation), evaluation=sorted(evaluation))
        if task_name in splits and splits[task_name] != split:
            raise ValueError('Panel-dependent question splits would leak labels')
        splits[task_name] = split
        matched = set(audit['matched_ids'])
        if matched != set(paper_cell['matched_ids']) or not matched <= evaluation:
            raise ValueError('Debate report mask differs from composition audit')
        initial, current = [], []
        for alias in aliases:
            source = settings['round0'][alias]
            path = portable_source(project, source['path'])
            checked(path, source['sha256'])
            initial.append(records(path))
            current.append(records(directory / 'round_1' / f'{alias}.jsonl'))
        if any(set(r) != fit | evaluation for r in initial + current):
            raise ValueError('Debate or initial records do not cover the audited split exactly')
        config = audit['config']
        missing, counts, group_rows = [], Counter(), []
        for qid in sorted(saved):
            before, after = [r[qid] for r in initial], [r[qid] for r in current]
            first = before[0]
            for m in before + after:
                if any(m[k] != first[k] for k in ('question', 'gold', 'options', 'choices')):
                    raise ValueError(f'{task_name}/{qid}: question/gold/option mapping mismatch')
                expected = m['prediction'] is not None and task.equivalent(str(m['prediction']), str(m['gold']))
                if bool(m['correct']) != expected:
                    raise ValueError(f'{task_name}/{qid}: correctness mismatch')
            for alias, m in zip(aliases, after):
                if m['round'] != 1 or m['model'] != alias or m['group'] != aliases:
                    raise ValueError('Turn metadata differs from panel')
                if sorted(m['peer_order'] or []) != sorted(a for a in aliases if a != alias):
                    raise ValueError('Debate trace is not full peer exchange')
            target, answers, reason = select_vote(task, after, model_ids, qid, config['samples'], config['tie_break'], config['tie_seed'])
            target0, answers0, reason0 = select_vote(task, before, model_ids, qid, config['samples'], config['tie_break'], config['tie_seed'])
            record = saved[qid]
            for stage, value, why in (('post_debate', target, reason), ('initial', target0, reason0)):
                expected = record[stage]['target']
                if not (value == expected or (value is not None and expected is not None and task.equivalent(value, expected))):
                    raise ValueError(f'{task_name}/{models}/{qid}: {stage} target differs')
                if why != record[stage]['selection_reason']:
                    raise ValueError(f'{task_name}/{models}/{qid}: {stage} selection reason differs')
            correctness = int(target is not None and task.equivalent(target, str(first['gold'])))
            if correctness != int(record['correct']) or record['split'] != ('fitting' if qid in fit else 'evaluation'):
                raise ValueError('Saved correctness or split differs from reconstruction')
            if any(m['prediction'] is None or m['mean_logprob'] is None or m.get('error') for m in before + after):
                missing.append(qid)
                continue
            name = 'train' if qid in train else 'validation' if qid in validation else 'evaluation'
            row = dict(task=task_name, models=models, model_ids=model_ids, id=qid, protocol='debate', round=1,
                       question=first['question'], target=target, answers=answers, selection_reason=reason,
                       mean_logprobs=[m['mean_logprob'] for m in after], split=name,
                       member_correct=[int(m['correct']) for m in after], correct=correctness,
                       initial_target=target0, initial_answers=answers0,
                       initial_mean_logprobs=[m['mean_logprob'] for m in before],
                       initial_member_correct=[int(m['correct']) for m in before],
                       communication=[[int(i != j) for j in range(len(aliases))] for i in range(len(aliases))])
            rows.append(row); group_rows.append(row); counts[name] += 1
        if matched & set(missing):
            raise ValueError(f'{task_name}/{models}: {len(matched & set(missing))} matched examples lack CAGE inputs; refusing to change the mask')
        vote_accuracy = sum(r['correct'] for r in group_rows if r['id'] in matched) / len(matched)
        cells.append(dict(paper_cell, estimator=sample['estimator'], coverage=len(matched)/len(evaluation),
                          vote_accuracy=vote_accuracy,
                          accuracy_all=sum(int(saved[q]["correct"]) for q in evaluation)/len(evaluation)))
        groups.append(dict(task=task_name, models=models, round=1, counts=dict(counts), missing_feature_ids=missing,
                           tie_break=config['tie_break'], tie_seed=config['tie_seed'], n_samples=config['samples']))
    return dict(project=project, protocol='debate', rows=rows, cells=cells, splits=splits,
                sources=provenance, groups=groups, atomic_path=table_dir / 'atomic.csv')
