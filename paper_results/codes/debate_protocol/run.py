import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))

from conf_compose.utils import metrics
from conf_compose.utils.calibration import fit_temperature, temperature_scale

TASKS = ('csqa', 'boolq', 'gsm8k', 'truthfulqa', 'gpqa')
SIZES = (2, 3, 4, 5, 6)
RAW_FIELDS = ('accuracy', 'ece', 'auarc', 'auroc', 'brier', 'nll')
FIELDS = (*RAW_FIELDS, 't_brier', 't_ece')
RULES = {
    'reference_stream_target': 'Single stream, same answer',
    'mean': 'Arithmetic mean',
    'logodds_sum': 'Log-odds sum',
    'logodds_mean': 'Log-odds mean',
    'shared_rho': 'Shared rho',
    'shared_scale': 'Shared scale',
    'kahn': 'Kahn (full covariance)',
    'blp': 'Weighted BLP',
    'logistic_pool': 'Regularized logistic pooling',
    'blp_equal': 'Equal-weight BLP',
    'kahn_diagonal': 'Kahn (diagonal covariance)',
}
LOWER = {'ece', 't_ece', 'brier', 't_brier', 'nll'}
NAMES = {'accuracy': 'Acc', 'ece': 'ECE', 't_ece': 't-ECE', 'auarc': 'AUARC', 'auroc': 'AUROC',
         'brier': 'Brier', 't_brier': 't-Brier', 'nll': 'NLL'}


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            value.update(chunk)
    return value.hexdigest()


def read_csv(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def valid(value):
    if value in ('', None):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def key(row):
    return row['task'], row['models'], int(row['round'])


def scores(probabilities, labels):
    if not probabilities:
        return {field: None for field in RAW_FIELDS}
    result = {'accuracy': mean(labels)}
    for field in RAW_FIELDS[1:]:
        value = float(getattr(metrics, field)(probabilities, labels))
        result[field] = value if math.isfinite(value) else None
    return result


def reference_stream(audit, example_path):
    with example_path.open() as handle:
        records = {record['id']: record for record in (json.loads(line) for line in handle)}
    if len(records) != len(audit['fit_ids']) + len(audit['eval_ids']):
        raise ValueError(f'duplicate or missing saved predictions: {example_path}')
    matched = audit['matched_ids']
    models = audit['models'].split('|')
    index = models.index(audit['reference_model'])
    if set(matched) - set(audit['eval_ids']):
        raise ValueError(f'matched IDs are outside the evaluation split: {example_path}')
    if set(records) != set(audit['fit_ids']) | set(audit['eval_ids']):
        raise ValueError(f'prediction IDs differ from audit: {example_path}')
    if any(not records[question]['matched'] for question in matched):
        raise ValueError(f'question mask disagrees with predictions: {example_path}')
    correct = [float(records[question]['correct']) for question in matched]
    shared = scores([records[question]['post_debate']['shared'][index] for question in matched], correct)
    return {'method': 'reference_stream_target', 'status': 'fixed', 'estimator': audit['estimator'],
            'n_models': audit['n_models'], 'n_scored': len(matched),
            'coverage': len(matched) / audit['n_evaluation'], 'accuracy_all': audit['pool_accuracy_all'],
            **shared}


def temperature_metrics(records, audit, method):
    index = audit['models'].split('|').index(audit['reference_model'])

    def confidence(question):
        record = records[question]
        if method == 'reference_stream_target':
            return record['post_debate']['shared'][index]
        prediction = record['pooled'].get(method)
        return prediction['confidence'] if prediction is not None else None

    fitting = [(confidence(question), float(records[question]['correct'])) for question in audit['fit_scored_ids']]
    fitting = [(value, label) for value, label in fitting if value is not None]
    evaluation = [(confidence(question), float(records[question]['correct'])) for question in audit['matched_ids']]
    evaluation = [(value, label) for value, label in evaluation if value is not None]
    if not fitting or not evaluation:
        return {'t_brier': None, 't_ece': None}, None, len(evaluation)
    temperature = fit_temperature(*zip(*fitting))
    probabilities = temperature_scale([value for value, _ in evaluation], temperature)
    labels = [label for _, label in evaluation]
    return {'t_brier': metrics.brier(probabilities, labels),
            't_ece': metrics.ece(probabilities, labels)}, temperature, len(evaluation)


def prepare(run):
    manifest = json.loads((run / 'manifest.json').read_text())
    if manifest['identity']['estimators'] != ['cons'] or manifest['skipped']:
        raise ValueError('report needs one completed consistency-only run without skipped cells')
    for name, expected in manifest['artifacts'].items():
        if digest(run / name) != expected:
            raise ValueError(f'composition artifact changed: {name}')
    source = read_csv(run / 'metrics.csv')
    by_cell = {}
    for row in source:
        by_cell.setdefault(key(row), []).append(row)
    if len(by_cell) != manifest['n_cells'] or len(source) != manifest['n_metric_rows']:
        raise ValueError('composition cells or metric rows are incomplete')
    expected = {(cell['task'], tuple(cell['models']), cell['round']) for cell in manifest['identity']['cells']}
    observed = {(task, tuple(models.split('|')), round_index) for task, models, round_index in by_cell}
    if expected != observed:
        raise ValueError('composition cells do not match manifest')
    rows, audits = [], []
    for identity, group in sorted(by_cell.items()):
        task, models, round_index = identity
        name = group[0]['cell']
        path = run / task / name / f'round_{round_index}' / 'cons'
        audit = json.loads((path / 'audit.json').read_text())
        if (audit['task'], audit['models'], audit['round'], audit['estimator']) != (task, models, round_index, 'cons'):
            raise ValueError(f'audit identity mismatch: {path}')
        if audit['n_matched'] != len(audit['matched_ids']) or audit['n_evaluation'] != len(audit['eval_ids']):
            raise ValueError(f'audit counts mismatch: {path}')
        present = {row['method'] for row in group}
        if round_index != 1:
            raise ValueError(f'only round-one debate is supported: {path}')
        if not ({'best_solo'} | (RULES.keys() - {'reference_stream_target'})) <= present:
            raise ValueError(f'missing composition methods: {path}')
        source_reference = next(row for row in group if row['method'] == 'best_solo')
        reference = reference_stream(audit, path / 'predictions.jsonl')
        with (path / 'predictions.jsonl').open() as handle:
            predictions = {record['id']: record for record in (json.loads(line) for line in handle)}
        temperatures = {}
        for method in RULES:
            values, temperature, n_scored = temperature_metrics(predictions, audit, method)
            source_row = reference if method == 'reference_stream_target' else next(row for row in group if row['method'] == method)
            if n_scored != int(source_row['n_scored']):
                raise ValueError(f'temperature mask differs from saved method mask: {path}/{method}')
            temperatures[method] = (values, temperature)
        reference.update(temperatures['reference_stream_target'][0])
        audits.append({'task': task, 'models': models, 'round': round_index, 'n_models': audit['n_models'],
                       'n_evaluation': audit['n_evaluation'], 'n_matched': audit['n_matched'],
                       'matched_ids': audit['matched_ids'],
                       'reference_model': audit['reference_model'], 'reference_mode': audit['reference_mode'],
                       'confidence': 'consistency',
                       'excluded': audit['excluded'], 'selection_reasons': audit['selection_reasons']})
        for row in [*group, reference]:
            if row['method'] not in RULES or row['estimator'] != 'cons':
                continue
            parsed = {'task': task, 'models': models, 'n_models': int(row['n_models']), 'round': round_index,
                      'estimator': 'cons', 'method': row['method'], 'status': row['status'],
                      'n_evaluation': audit['n_evaluation'], 'n_matched': audit['n_matched'],
                      'n_scored': int(row['n_scored']), 'coverage': valid(row['coverage']),
                      'reference_model': audit['reference_model'], 'reference_mode': audit['reference_mode'],
                      'cell': name, 'output_temperature': temperatures[row['method']][1]}
            if parsed['n_scored'] > parsed['n_matched']:
                raise ValueError(f'method scored more than common mask: {path}')
            for field in FIELDS:
                value = (temperatures[row['method']][0][field] if field.startswith('t_') else valid(row[field]))
                reference_value = valid(reference[field])
                if field in RAW_FIELDS and row is not reference and valid(row['reference_' + field]) != valid(source_reference[field]):
                    raise ValueError(f'reference differs across methods: {path}/{row["method"]}/{field}')
                parsed[field] = valid(row['accuracy_all']) if field == 'accuracy' else value
                parsed['reference_' + field] = valid(reference['accuracy_all']) if field == 'accuracy' else reference_value
                parsed['delta_' + field] = (parsed[field] - parsed['reference_' + field]
                                            if parsed[field] is not None and parsed['reference_' + field] is not None else None)
            rows.append(parsed)
    return rows, audits, manifest


def summary(rows, audits, method, task, size, field, comparison=None):
    selected = [row for row in rows if row['method'] == method and row['task'] == task
                and (size is None or row['n_models'] == size)]
    expected = sum(audit['task'] == task and (size is None or audit['n_models'] == size) for audit in audits)
    value = ('delta_' + field) if comparison == 'target' else field
    if comparison == 'mean':
        baseline = 'mean'
        baselines = {(row['task'], row['models'], row['round']): row[field]
                     for row in rows if row['method'] == baseline}
        observed = [row[field] - baselines[row['task'], row['models'], row['round']] for row in selected
                    if row[field] is not None and baselines[row['task'], row['models'], row['round']] is not None]
    else:
        observed = [row[value] for row in selected if row[value] is not None]
    return mean(observed) if observed else None, stdev(observed) if len(observed) > 1 else None, len(observed), expected


def display(value, deviation, field, delta):
    if value is None:
        return '--'
    scale = 1 if field == 'nll' else 100
    digits = 3 if field == 'nll' else 2
    number = f'{value * scale:+.{digits}f}' if delta else f'{value * scale:.{digits}f}'
    return number + (f' ± {deviation * scale:.{digits}f}' if delta and deviation is not None else '')


def escape(value):
    return str(value).replace('\\', r'\textbackslash{}').replace('_', r'\_').replace('&', r'\&')


def table(rows, audits, tasks, size, fields, comparison):
    methods = list(RULES)
    title = {None: 'Absolute results',
             'target': 'Delta from single stream on the same answer',
             'mean': 'Delta from arithmetic mean'}[comparison]
    scope = 'all 15 model groups' if size is None else f'groups of {size} models'
    notes = [f'Post-debate consistency composition, round 1; {scope}; {title}.',
             'Each dataset/model group has equal weight. Differences are calculated per group before averaging.',
             'Acc uses every evaluation question; confidence metrics use the group\'s common scored-question mask.',
             'ECE, t-ECE, AUARC, AUROC, Acc, Brier and t-Brier are x100; NLL is unscaled (nats).',
             'Positive delta AUARC/AUROC/Acc and negative delta ECE/Brier/NLL are improvements.',
             '± is sample SD across model groups, not a confidence interval. See coverage tables for group counts.',
             'Every method scores the same selected debate answer on the same questions.', '']
    columns = [(task, field) for task in tasks for field in fields]
    headers = ['Method'] + [f'{task.upper()} {field.upper()}' for task, field in columns]
    grid = [[RULES[method]] + [display(*summary(rows, audits, method, task, size, field, comparison)[:2], field, comparison is not None)
                               for task, field in columns] for method in methods]
    widths = [max(len(str(row[i])) for row in [headers, *grid]) for i in range(len(headers))]
    lines = [' | '.join(str(value).ljust(width) for value, width in zip(row, widths)) for row in [headers, *grid]]
    lines.insert(1, '-+-'.join('-' * width for width in widths))
    txt = '\n'.join([*notes, *lines, ''])
    latex = ['% ' + note for note in notes if note]
    latex += [r'\begin{tabular}{l' + 'r' * len(columns) + '}', r'\toprule',
              'Method & ' + ' & '.join(rf'\multicolumn{{{len(fields)}}}{{c}}{{{escape(task.upper())}}}' for task in tasks) + r' \\',
              ' & ' + ' & '.join((r'$\Delta$ ' if comparison else '') + field.upper()
                                   + (r' $\downarrow$' if field in LOWER else r' $\uparrow$')
                                   for task, field in columns) + r' \\', r'\midrule']
    for method in methods:
        cells = []
        for task, field in columns:
            value, deviation, _, _ = summary(rows, audits, method, task, size, field, comparison)
            cell = display(value, deviation, field, comparison is not None)
            if deviation is not None and comparison is not None:
                left, right = cell.split(' ± ')
                cell = left + r' {\scriptsize $\pm$ ' + right + '}'
            cells.append(cell)
        latex.append(escape(RULES[method]) + ' & ' + ' & '.join(cells) + r' \\')
    latex += [r'\bottomrule', r'\end{tabular}', '']
    return txt, '\n'.join(latex)


def sign_test(rows, audits, tasks):
    tests = []
    for method in RULES:
        if method == 'reference_stream_target':
            continue
        for field in ('ece', 'auarc'):
            deltas = []
            for task in tasks:
                value, _, n, total = summary(rows, audits, method, task, None, field, 'target')
                if n == total and value is not None:
                    deltas.append(value * (-1 if field == 'ece' else 1))
            nonzero = [value for value in deltas if abs(value) > 1e-12]
            wins = sum(value > 0 for value in nonzero)
            count = len(nonzero)
            p = sum(math.comb(count, j) for j in range(wins, count + 1)) / 2 ** count
            tests.append({'method': method, 'metric': field, 'datasets': len(deltas), 'wins': wins,
                          'non_tied': count, 'p_raw': p})
    ordering = sorted(range(len(tests)), key=lambda i: tests[i]['p_raw'])
    previous = 0.0
    for rank, index in enumerate(ordering):
        previous = max(previous, min(1.0, (len(tests) - rank) * tests[index]['p_raw']))
        tests[index]['p_holm'] = previous
    return tests


def paper_highlights(rows, audits, sections, tasks, fields):
    highlights = {}
    macros = ('gfirst', 'gsecond', 'gthird')
    names = [name for _, section in sections for name in section]
    for task in tasks:
        for field in fields:
            candidates = []
            for name in names:
                value = summary(rows, audits, name, task, None, field)[0]
                if value is not None:
                    candidates.append((name, value))
            candidates.sort(key=lambda item: item[1], reverse=field == 'auarc')
            rank = 0
            previous = None
            for name, value in candidates:
                if value is None:
                    continue
                if previous is None or abs(value - previous) > 1e-12:
                    rank += 1
                    previous = value
                if rank > len(macros):
                    break
                highlights[name, task, field] = macros[rank - 1]
    return highlights


def paper_table(rows, audits, tasks):
    sections = (
        ('Reference', ('reference_stream_target',)),
        ('No learned combination weights', ('mean', 'logodds_sum')),
        ('Learned per-model weights', ('kahn', 'kahn_diagonal', 'blp', 'logistic_pool')),
    )
    fields = ('brier', 't_brier', 'auarc')
    highlights = paper_highlights(rows, audits, sections, tasks, fields)
    width = 1 + len(tasks) * len(fields)
    lines = [r'\begin{table*}[t]', r'\centering',
             r'\caption{Confidence quality after one debate round. Absolute metrics are averaged across available model groups (15 per dataset, except one zero-coverage logistic-pooling group on BoolQ); entries are mean $\pm$ sample SD. All methods score the same selected answer.}',
             r'\label{tab:debate-consistency-absolute}',
             r'\setlength{\tabcolsep}{2.6pt}', r'\resizebox{\textwidth}{!}{%',
             r'\begin{tabular}{@{}l' + 'r' * (width - 1) + r'@{}}', r'\toprule',
             ' & ' + ' & '.join(rf'\multicolumn{{3}}{{c}}{{{escape(task.upper())}}}' for task in tasks) + r' \\',
             'Method & ' + ' & '.join(('Brier $\\downarrow$', 't-Brier $\\downarrow$', 'AUARC $\\uparrow$') * len(tasks)) + r' \\',
             r'\midrule']
    for section, methods in sections:
        lines.append(rf'\multicolumn{{{width}}}{{l}}{{\textbf{{{section}}}}}\\')
        for method in methods:
            cells = []
            for task in tasks:
                for field in fields:
                    value, deviation, _, _ = summary(rows, audits, method, task, None, field)
                    cell = '--' if value is None else f'{value:.3f}' + (
                        rf' {{\scriptsize $\pm$ {deviation:.3f}}}' if deviation is not None else ''
                    )
                    macro = highlights.get((method, task, field))
                    cells.append(rf'\{macro}{{{cell}}}' if macro else cell)
            lines.append(escape(RULES[method]) + ' & ' + ' & '.join(cells) + r' \\')
        lines.append(r'\midrule')
    lines[-1] = r'\bottomrule'
    lines += [r'\end{tabular}%', '}', r'\vspace{2pt}',
              r'\parbox{\textwidth}{\footnotesize t-Brier applies one output temperature fitted by NLL on the fitting split and held fixed on evaluation. AUARC uses the original ranking. Lower Brier and t-Brier, and higher AUARC, are better.}',
              r'\end{table*}']
    return '\n'.join(lines) + '\n'


def run(args):
    rows, audits, source = prepare(args.run)
    tasks = [task for task in TASKS if task in {row['task'] for row in rows}]
    if len(tasks) != len({row['task'] for row in rows}):
        raise ValueError('unknown dataset in composition output')
    output = args.out / ('run_' + source['run_id']) / 'cons'
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'atomic.csv', rows)
    write_csv(output / 'significance.csv', sign_test(rows, audits, tasks))
    (output / 'audit.json').write_text(json.dumps(audits, indent=2) + '\n')
    for size in (None, *SIZES):
        prefix = 'all' if size is None else f'size_{size}'
        for tag, fields in (('', ('ece', 'auarc')), ('_full', FIELDS), ('_accuracy', ('accuracy',))):
            for comparison, suffix in ((None, 'absolute'), ('target', 'delta'), ('mean', 'vs_mean')):
                txt, tex = table(rows, audits, tasks, size, fields, comparison)
                for extension, content in (('txt', txt), ('tex', tex)):
                    (output / f'{prefix}{tag}_{suffix}.{extension}').write_text(content)
        coverage = ['Method' + ''.join(f' | {task.upper()}' for task in tasks)]
        for method, label in RULES.items():
            coverage.append(label + ''.join(f' | {summary(rows, audits, method, task, size, "ece")[2]}/'
                                            f'{summary(rows, audits, method, task, size, "ece")[3]}' for task in tasks))
        (output / f'{prefix}_coverage.txt').write_text('\n'.join(coverage) + '\n')
    if args.out.resolve() == (ROOT / 'paper_results/results/debate_protocol').resolve():
        (ROOT / 'paper_results/z_paper_tables/debate_consistency.tex').write_text(paper_table(rows, audits, tasks))
    provenance = {'source_run': str(args.run.resolve()), 'source_run_id': source['run_id'],
                  'source_manifest_sha256': digest(args.run / 'manifest.json'),
                  'source_metrics_sha256': digest(args.run / 'metrics.csv'),
                  'report_code_sha256': digest(Path(__file__)), 'tasks': tasks,
                  'groups_per_task': {task: sum(audit['task'] == task for audit in audits) for task in tasks},
                  'rounds': sorted({row['round'] for row in rows}), 'estimator': 'cons',
                  'n_rows': len(rows),
                  'significance': 'exploratory one-sided dataset-block sign tests with joint Holm correction',
                  'accuracy': 'all evaluation questions; confidence metrics use each group common matched subset'}
    provenance['temperature_calibration'] = 'one output temperature fitted by NLL on original fitting questions; AUARC unchanged'
    (output / 'manifest.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(f'Wrote {len(rows)} method/group rows across {len(audits)} groups to {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Report saved post-debate consistency composition')
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=ROOT / 'paper_results/results/debate_protocol')
    run(parser.parse_args())
