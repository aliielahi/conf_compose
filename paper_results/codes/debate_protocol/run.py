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
REPORT_FIELDS = (*FIELDS, 'answer_matched_auarc')
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
CAGE_RUN = ROOT / 'baselines/results/debate_adapter/71259088b2f0e984'
CAGE_RULES = {'cagecal_debate': 'CAGE-CAL (paired debate)',
              'cagecal_debate_betasb': 'CAGE-CAL (paired debate + BetaSB)'}
REPORT_RULES = {**RULES, **CAGE_RULES}
LOWER = {'ece', 't_ece', 'brier', 't_brier', 'nll'}
NAMES = {'accuracy': 'Acc', 'ece': 'ECE', 't_ece': 't-ECE', 'auarc': 'AUARC', 'auroc': 'AUROC',
         'brier': 'Brier', 't_brier': 't-Brier', 'nll': 'NLL',
         'answer_matched_auarc': 'AUARC (same answer)'}


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


def prepare(run, reference_mode='fit_accuracy'):
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
        with (path / 'predictions.jsonl').open() as handle:
            predictions = {record['id']: record for record in (json.loads(line) for line in handle)}
        if reference_mode == 'fit_auarc':
            fitting = audit['fit_scored_ids']
            if not fitting:
                raise ValueError(f'no fitting scores for AUARC reference: {path}')
            labels = [float(predictions[question]['correct']) for question in fitting]
            candidate_scores = [scores([predictions[question]['post_debate']['shared'][index] for question in fitting], labels)['auarc']
                                for index in range(len(models.split('|')))]
            if any(value is None for value in candidate_scores):
                raise ValueError(f'undefined fitting AUARC reference: {path}')
            audit['reference_model'] = models.split('|')[max(range(len(candidate_scores)), key=lambda index: candidate_scores[index])]
        audit['reference_mode'] = reference_mode
        reference = reference_stream(audit, path / 'predictions.jsonl')
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
            parsed['answer_matched_auarc'] = parsed['auarc']
            parsed['reference_answer_matched_auarc'] = parsed['reference_auarc']
            parsed['delta_answer_matched_auarc'] = parsed['delta_auarc']
            rows.append(parsed)
    return rows, audits, manifest


def append_cagecal(rows, audits, directory):
    directory = Path(directory)
    source_audits = json.loads((directory / 'paper_tables/audit.json').read_text())
    source_cells = {(cell['task'], cell['models'], int(cell.get('round', 1))): cell for cell in source_audits}
    with (directory / 'paper_tables/cagecal_metrics.csv').open(newline='') as handle:
        source_metrics = {(row['task'], row['models'], row['method']): row for row in csv.DictReader(handle)}

    def load(path):
        found = {}
        with path.open() as handle:
            for line in handle:
                item = json.loads(line)
                identity = (item['task'], item['models'], item['id'])
                if identity in found:
                    raise ValueError(f'duplicate CAGE-CAL prediction: {identity}')
                found[identity] = item
        return found

    predictions = load(directory / 'predictions.jsonl')
    validation = load(directory / 'validation_predictions.jsonl')
    references = {(row['task'], row['models'], row['round']): row for row in rows
                  if row['method'] == 'reference_stream_target'}
    if set(references) != set(source_cells):
        raise ValueError('CAGE-CAL debate cells do not match composition cells')
    for audit in audits:
        identity = (audit['task'], audit['models'], audit['round'])
        source = source_cells[identity]
        if source['matched_ids'] != audit['matched_ids'] or source['n_matched'] != audit['n_matched']:
            raise ValueError(f'CAGE-CAL debate question mask differs: {identity}')
        selected = [predictions[(audit['task'], audit['models'], question)] for question in audit['matched_ids']]
        labels = [float(item['correct']) for item in selected]
        if abs(mean(labels) - source['vote_accuracy']) > 1e-12:
            raise ValueError(f'CAGE-CAL debate labels differ: {identity}')
        val = [item for (task, models, _), item in validation.items()
               if (task, models) == identity[:2]]
        if not val or set(item['id'] for item in val) & set(audit['matched_ids']):
            raise ValueError(f'CAGE-CAL validation missing or overlapping: {identity}')
        for method, field in (('cagecal_debate', 'raw'), ('cagecal_debate_betasb', 'betasb')):
            export = source_metrics[audit['task'], audit['models'], method]
            probabilities = [float(item[field]) for item in selected]
            observed = scores(probabilities, labels)
            for metric in RAW_FIELDS:
                expected = valid(export[metric])
                actual = observed[metric]
                if metric == 'accuracy':
                    continue
                if actual is not None and expected is not None and abs(actual - expected) > 1e-9:
                    raise ValueError(f'CAGE-CAL debate {metric} differs: {identity}/{method}')
            temperature = None
            calibrated = {'t_brier': None, 't_ece': None}
            if method == 'cagecal_debate':
                temperature = (fit_temperature([item['raw'] for item in val], [item['correct'] for item in val])
                               if len({item['correct'] for item in val}) > 1 else 1.0)
                if abs(temperature - float(export['output_temperature'])) > 1e-5:
                    raise ValueError(f'CAGE-CAL debate temperature differs: {identity}')
                temperature = float(export['output_temperature'])
                scaled = temperature_scale(probabilities, temperature)
                calibrated = {'t_brier': float(metrics.brier(scaled, labels)),
                              't_ece': float(metrics.ece(scaled, labels))}
                for metric, value in calibrated.items():
                    if abs(value - float(export[metric])) > 1e-9:
                        raise ValueError(f'CAGE-CAL debate {metric} differs: {identity}')
            row = dict(references[identity], method=method, status='trained', output_temperature=temperature)
            for metric, value in {**observed, **calibrated}.items():
                if metric == 'accuracy':
                    continue
                baseline = row['reference_' + metric]
                row[metric] = value
                row['delta_' + metric] = value - baseline if value is not None and baseline is not None else None
            row['answer_matched_auarc'] = row['auarc']
            row['delta_answer_matched_auarc'] = row['delta_auarc']
            rows.append(row)
    return {'directory': str(directory.resolve()), 'cells': len(source_cells),
            'validation_predictions_sha256': digest(directory / 'validation_predictions.jsonl'),
            'evaluation_predictions_sha256': digest(directory / 'predictions.jsonl')}


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
    methods = list(REPORT_RULES)
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
    grid = [[REPORT_RULES[method]] + [display(*summary(rows, audits, method, task, size, field, comparison)[:2], field, comparison is not None)
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
        latex.append(escape(REPORT_RULES[method]) + ' & ' + ' & '.join(cells) + r' \\')
    latex += [r'\bottomrule', r'\end{tabular}', '']
    return txt, '\n'.join(latex)


def sign_test(rows, audits, tasks):
    tests = []
    for method in REPORT_RULES:
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
            candidates.sort(key=lambda item: item[1], reverse=field in ('auarc', 'answer_matched_auarc', 'delta_answer_matched_auarc'))
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
        ('External baseline', tuple(CAGE_RULES)),
        ('Learned per-model weights', ('kahn', 'kahn_diagonal', 'blp', 'logistic_pool')),
    )
    fields = ('t_ece', 't_brier', 'auarc', 'delta_answer_matched_auarc')
    highlights = paper_highlights(rows, audits, sections, tasks, fields)
    width = 1 + len(tasks) * len(fields)
    lines = [r'\begin{table*}[t]', r'\centering',
             r'\caption{Confidence quality after one debate round. Absolute metrics are averaged across available model groups (15 per dataset, except one zero-coverage logistic-pooling group on BoolQ); entries are mean $\pm$ sample SD. All methods score the same selected answer.}',
             r'\label{tab:debate-consistency-absolute}',
             r'\setlength{\tabcolsep}{2.6pt}', r'\resizebox{\textwidth}{!}{%',
             r'\begin{tabular}{@{}l' + 'r' * (width - 1) + r'@{}}', r'\toprule',
             ' & ' + ' & '.join(rf'\multicolumn{{4}}{{c}}{{{escape(task.upper())}}}' for task in tasks) + r' \\',
             'Method & ' + ' & '.join(('t-ECE $\\downarrow$', 't-Brier $\\downarrow$', 'AUARC $\\uparrow$', '$\\Delta$AUARC (same answer) $\\uparrow$') * len(tasks)) + r' \\',
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
            lines.append(escape(REPORT_RULES[method]) + ' & ' + ' & '.join(cells) + r' \\')
        lines.append(r'\midrule')
    lines[-1] = r'\bottomrule'
    lines += [r'\end{tabular}%', '}', r'\vspace{2pt}',
              r'\parbox{\textwidth}{\footnotesize t-ECE and t-Brier use a validation-fitted output temperature on raw scores. AUARC uses original rankings; $\Delta$AUARC compares with one model scoring the same final answer. CAGE-CAL BetaSB has no additional temperature fit.}',
              r'\end{table*}']
    return '\n'.join(lines) + '\n'


def run(args):
    rows, audits, source = prepare(args.run, args.reference)
    cage_provenance = append_cagecal(rows, audits, args.cagecal_dir)
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
        for tag, fields in (('', ('ece', 'auarc')), ('_full', REPORT_FIELDS), ('_accuracy', ('accuracy',))):
            for comparison, suffix in ((None, 'absolute'), ('target', 'delta'), ('mean', 'vs_mean')):
                txt, tex = table(rows, audits, tasks, size, fields, comparison)
                for extension, content in (('txt', txt), ('tex', tex)):
                    (output / f'{prefix}{tag}_{suffix}.{extension}').write_text(content)
        coverage = ['Method' + ''.join(f' | {task.upper()}' for task in tasks)]
        for method, label in REPORT_RULES.items():
            coverage.append(label + ''.join(f' | {summary(rows, audits, method, task, size, "ece")[2]}/'
                                            f'{summary(rows, audits, method, task, size, "ece")[3]}' for task in tasks))
        (output / f'{prefix}_coverage.txt').write_text('\n'.join(coverage) + '\n')
    if args.out.resolve() == (ROOT / 'paper_results/results/debate_protocol').resolve() and args.reference == 'fit_accuracy':
        (ROOT / 'paper_results/z_paper_tables/debate_consistency.tex').write_text(paper_table(rows, audits, tasks))
    provenance = {'source_run': str(args.run.resolve()), 'source_run_id': source['run_id'],
                  'source_manifest_sha256': digest(args.run / 'manifest.json'),
                  'source_metrics_sha256': digest(args.run / 'metrics.csv'),
                  'report_code_sha256': digest(Path(__file__)), 'tasks': tasks,
                  'groups_per_task': {task: sum(audit['task'] == task for audit in audits) for task in tasks},
                  'rounds': sorted({row['round'] for row in rows}), 'estimator': 'cons',
                  'n_rows': len(rows), 'reference': args.reference, 'cagecal': cage_provenance,
                  'significance': 'exploratory one-sided dataset-block sign tests with joint Holm correction',
                  'accuracy': 'all evaluation questions; confidence metrics use each group common matched subset'}
    provenance['temperature_calibration'] = 'one output temperature fitted by NLL on original fitting questions; AUARC unchanged'
    (output / 'manifest.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(f'Wrote {len(rows)} method/group rows across {len(audits)} groups to {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Report saved post-debate consistency composition')
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=ROOT / 'paper_results/results/debate_protocol')
    parser.add_argument('--reference', choices=('fit_accuracy', 'fit_auarc'), default='fit_accuracy')
    parser.add_argument('--cagecal-dir', type=Path, default=CAGE_RUN)
    run(parser.parse_args())
