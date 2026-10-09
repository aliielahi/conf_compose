import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np

from conf_compose.data import get_task
from conf_compose.utils import metrics

from .candidates import candidate_set, label_space, support
from .evidence import Stream
from .methods import majority_answer
from .pooling import FIXED_RULES, fit_methods, pool_methods

METRICS = ('accuracy', 'ece', 'auarc', 'auroc', 'brier', 'nll')
SAMPLE_KEY = 'consistency_t0.7'
# Pools every member's round-0 and post-debate confidence in the final answer: 2N streams instead of N.
BOTH_ROUNDS = 'both_rounds:'


@dataclass(frozen=True)
class CompositionConfig:
    samples: int = 5
    fit_fraction: float = 0.3
    fit_seed: int = 0
    tie_break: str = 'confidence'
    tie_seed: int = 0
    context: str = 'direct'
    seq_score: str = 'norm_sum'
    ablations: bool = True
    logistic_l2: float = 1.0

    def __post_init__(self):
        if self.samples < 1 or not 0 < self.fit_fraction < 1 or self.logistic_l2 < 0:
            raise ValueError('invalid sample count, fitting fraction or regularization')
        if self.tie_break not in ('confidence', 'first') or self.context not in ('direct', 'reasoned'):
            raise ValueError('invalid tie-break or scoring context')
        if self.seq_score not in ('norm_sum', 'norm_mean'):
            raise ValueError('invalid sequence score')


def id_digest(ids):
    return hashlib.sha256(json.dumps(sorted(ids), separators=(',', ':')).encode()).hexdigest()


def split_ids(task, ids, config):
    fitting, evaluation = [], []
    for question in sorted(ids):
        key = json.dumps([task, config.fit_seed, question]).encode()
        position = int.from_bytes(hashlib.sha256(key).digest()[:8], 'big') / 2**64
        (fitting if position < config.fit_fraction else evaluation).append(question)
    if not fitting or not evaluation:
        raise ValueError('the saved questions must provide nonempty fitting and evaluation splits')
    return fitting, evaluation


def sequence_score(task, record, target, config):
    candidates = (record.get('candidate_scores') or {}).get('candidates') or []
    if target is None or not candidates or record.get('error'):
        return None
    labels = record.get('options') or label_space(task)
    if labels and any(not any(task.equivalent(label, c['answer']) for c in candidates) for label in labels):
        raise ValueError(f'incomplete closed-label candidate set: {record["id"]}')
    if len({candidate['answer'] for candidate in candidates}) != len(candidates):
        raise ValueError(f'duplicate sequence candidates: {record["id"]}')
    logs, selected = [], []
    for candidate in candidates:
        values = (candidate.get(config.context) or {}).get('logprobs')
        if not values or not all(math.isfinite(value) for value in values):
            return None
        value = sum(values)
        logs.append(value / len(values) if config.seq_score == 'norm_mean' else value)
        selected.append(task.equivalent(candidate['answer'], target))
    if not any(selected):
        return None
    weights = [math.exp(value - max(logs)) for value in logs]
    return sum(value for value, match in zip(weights, selected) if match) / sum(weights)


def score_question(task, models, records, question, round_index, estimator, config, toward=None):
    """`toward` scores this round's streams against another round's answer instead of their own majority."""
    streams = [Stream(f'{question}:r{round_index}:a{agent}', agent, round_index, f'vllm/{model}',
                      records[model][question]['prediction'],
                      (records[model][question].get('sampled_answers') or {}).get(SAMPLE_KEY, []))
               for agent, model in enumerate(models)]
    candidates = candidate_set(task, None, streams)
    supports = {stream.stream_id: support(task, stream, candidates, config.samples) for stream in streams}
    weights = {stream.stream_id: supports[stream.stream_id].binary(stream.answer)
               if len(stream.samples) >= config.samples else None for stream in streams}
    counts = {candidate: sum(stream.answer is not None and task.equivalent(stream.answer, candidate)
                             for stream in streams) for candidate in candidates}
    tied = [candidate for candidate in candidates if counts[candidate] == max(counts.values())]
    reason = 'no_answer' if not candidates else 'count'
    unavailable = any(weights[s.stream_id] is None for s in streams
                      if s.answer is not None and any(task.equivalent(s.answer, target) for target in tied))
    if len(tied) > 1:
        reason = 'first' if config.tie_break == 'first' else 'missing_consistency_seeded' if unavailable else 'confidence'
        if config.tie_break == 'confidence' and not unavailable:
            totals = [round(sum(weights[s.stream_id] for s in streams
                                if s.answer is not None and task.equivalent(s.answer, target)), 12) for target in tied]
            if totals.count(max(totals)) > 1:
                reason = 'equal_confidence_seeded'
    key = json.dumps([config.tie_seed, task.name, question, sorted(s.model for s in streams)])
    target = (majority_answer(task, streams) if config.tie_break == 'first'
              else majority_answer(task, streams, None if unavailable else weights, tie_key=key))

    def confidence(model, stream, answer):
        record = records[model][question]
        if record.get('error'):
            return None
        if estimator == 'seq':
            return sequence_score(task, record, answer, config)
        if len(stream.samples) < config.samples:
            return None
        return supports[stream.stream_id].binary(answer)

    goal = target if toward is None else toward
    return {'target': target, 'selection_reason': reason,
            'own': [confidence(model, stream, stream.answer) for model, stream in zip(models, streams)],
            'shared': [confidence(model, stream, goal) for model, stream in zip(models, streams)],
            'valid_samples': [supports[s.stream_id].valid for s in streams],
            'requested_samples': [min(len(s.samples), config.samples) for s in streams]}


def metric_values(probabilities, labels, ranking=None):
    if not probabilities:
        return {metric: None for metric in METRICS}
    probabilities = np.round(probabilities, 12)
    ranking = probabilities if ranking is None else np.round(ranking, 12)
    values = {'accuracy': float(np.mean(labels))}
    for metric in METRICS[1:]:
        values[metric] = float(getattr(metrics, metric)(ranking if metric in ('auroc', 'auarc') else probabilities, labels))
    return {key: value if math.isfinite(value) else None for key, value in values.items()}


def evaluate_cell(cell, estimator, config):
    if estimator not in ('cons', 'seq'):
        raise ValueError(f'unknown estimator: {estimator}')
    task, models = get_task(cell.task), cell.models
    fitting, evaluation = split_ids(cell.task, cell.ids, config)
    observations = {}
    for question in cell.ids:
        initial = score_question(task, models, cell.initial, question, 0, estimator, config)
        current = score_question(task, models, cell.current, question, cell.round, estimator, config)
        gold = cell.initial[models[0]][question]['gold']
        correct = current['target'] is not None and task.equivalent(current['target'], gold)
        initial_on_final = (score_question(task, models, cell.initial, question, 0, estimator, config,
                                           toward=current['target'])['shared']
                            if current['target'] is not None else [None] * len(models))
        observations[question] = {'id': question, 'gold': gold, 'initial': initial, 'post_debate': current,
                                  'initial_on_final': initial_on_final,
                                  'correct': bool(correct), 'scored': current['target'] is not None
                                  and all(value is not None for value in current['shared'])}
    fit_scored = [q for q in fitting if observations[q]['scored']]
    matrix = np.asarray([observations[q]['post_debate']['shared'] for q in fit_scored], dtype=float).reshape(-1, len(models))
    fitted = fit_methods(matrix, [observations[q]['correct'] for q in fit_scored],
                         ablations=config.ablations, logistic_l2=config.logistic_l2)
    rules = (*FIXED_RULES, *fitted)

    def both_rounds(question):
        """Post-debate then round-0 confidences in the final answer, or None if any stream is missing."""
        values = [*observations[question]['post_debate']['shared'], *observations[question]['initial_on_final']]
        return values if observations[question]['scored'] and all(v is not None for v in values) else None

    fit_both = [q for q in fitting if both_rounds(q) is not None]
    fitted_both = fit_methods(np.asarray([both_rounds(q) for q in fit_both], dtype=float).reshape(-1, 2 * len(models)),
                              [observations[q]['correct'] for q in fit_both],
                              ablations=config.ablations, logistic_l2=config.logistic_l2)
    rules_both = tuple(BOTH_ROUNDS + rule for rule in (*FIXED_RULES, *fitted_both))
    fitting_accuracy = {model: sum(bool(cell.initial[model][q]['correct']) for q in fitting) / len(fitting) for model in models}
    reference_model = max(models, key=fitting_accuracy.get)
    reference_index = models.index(reference_model)
    matched = [q for q in evaluation if observations[q]['scored']
               and all(value is not None for stage in ('initial', 'post_debate') for value in observations[q][stage]['own'])]
    matched_set, fitting_set = set(matched), set(fitting)
    predictions = []
    for question in cell.ids:
        observation = observations[question]
        pooled = {}
        if observation['scored']:
            shared = observation['post_debate']['shared']
            outputs = {**pool_methods(shared), **{name: fit.predict(shared) for name, fit in fitted.items()}}
            combined = both_rounds(question)
            if combined is not None:
                outputs.update({BOTH_ROUNDS + name: prediction for name, prediction in pool_methods(combined).items()})
                outputs.update({BOTH_ROUNDS + name: fit.predict(combined) for name, fit in fitted_both.items()})
            for name, prediction in outputs.items():
                rank = prediction.ranking_score
                if rank is None:
                    rank = prediction.logit if prediction.logit is not None else prediction.score
                pooled[name] = {'confidence': prediction.score, 'ranking': rank}
        predictions.append({**observation, 'split': 'fitting' if question in fitting_set else 'evaluation',
                            'matched': question in matched_set, 'pooled': pooled,
                            'initial_answers': [cell.initial[m][question]['prediction'] for m in models],
                            'post_answers': [cell.current[m][question]['prediction'] for m in models],
                            'initial_correct': [bool(cell.initial[m][question]['correct']) for m in models],
                            'post_correct': [bool(cell.current[m][question]['correct']) for m in models],
                            'errors': {m: cell.current[m][question]['error'] for m in models if cell.current[m][question].get('error')}})
    by_id = {row['id']: row for row in predictions}
    reference = metric_values([by_id[q]['initial']['own'][reference_index] for q in matched],
                              [cell.initial[reference_model][q]['correct'] for q in matched])
    base = {'task': cell.task, 'models': '|'.join(models), 'n_models': len(models), 'round': cell.round,
            'estimator': estimator, 'n_samples': config.samples, 'n_evaluation': len(evaluation),
            'n_matched': len(matched), 'reference_model': reference_model, 'reference_mode': 'initial_fit_accuracy'}
    rows = []

    def add_method(name, values, labels, ranking=None, status='available', all_accuracy=None):
        scores = metric_values(values, labels, ranking)
        row = {**base, 'method': name, 'status': status, 'n_scored': len(values),
               'coverage': len(values) / len(evaluation), 'accuracy_all': all_accuracy, **scores}
        for metric in METRICS:
            row[f'reference_{metric}'] = reference[metric]
            row[f'delta_{metric}'] = scores[metric] - reference[metric] if scores[metric] is not None and reference[metric] is not None else None
        rows.append(row)

    for stage, records in (('initial', cell.initial), ('post_debate', cell.current)):
        for i, model in enumerate(models):
            add_method(f'{stage}:{model}', [by_id[q][stage]['own'][i] for q in matched],
                       [records[model][q]['correct'] for q in matched],
                       all_accuracy=sum(bool(records[model][q]['correct']) for q in evaluation) / len(evaluation))
    reference_row = next(row for row in rows if row['method'] == f'initial:{reference_model}')
    rows.insert(0, {**reference_row, 'method': 'best_solo'})
    vote_accuracy = sum(by_id[q]['correct'] for q in evaluation) / len(evaluation)
    for rule in (*rules, *rules_both):
        valid = [q for q in matched if (by_id[q]['pooled'].get(rule) or {}).get('confidence') is not None]
        fit = fitted_both.get(rule[len(BOTH_ROUNDS):]) if rule.startswith(BOTH_ROUNDS) else fitted.get(rule)
        add_method(rule, [by_id[q]['pooled'][rule]['confidence'] for q in valid],
                   [by_id[q]['correct'] for q in valid], [by_id[q]['pooled'][rule]['ranking'] for q in valid],
                   fit.status if fit is not None else 'fixed', vote_accuracy)
    atomic = {**base, 'n_total': len(cell.ids), 'n_fit_questions': len(fitting), 'n_fit_scored': len(fit_scored),
              'n_fit_errors': sum(not by_id[q]['correct'] for q in fit_scored),
              'fit_ids_hash': id_digest(fitting), 'eval_ids_hash': id_digest(evaluation),
              'fit_scored_ids_hash': id_digest(fit_scored), 'eval_scored_ids_hash': id_digest(matched),
              'tie_break': config.tie_break, 'tie_seed': config.tie_seed,
              'context': config.context if estimator == 'seq' else '',
              'seq_score': config.seq_score if estimator == 'seq' else '',
              'vote_acc': vote_accuracy, 'vote_coverage': len(matched) / len(evaluation),
              'single_acc': json.dumps([next(r for r in rows if r['method'] == f'post_debate:{m}')['accuracy_all'] for m in models])}
    for stage, prefix in (('initial', 'initial_single'), ('post_debate', 'single')):
        for metric in METRICS:
            atomic[f'{prefix}_{metric}'] = json.dumps([next(r for r in rows if r['method'] == f'{stage}:{m}')[metric] for m in models])
    atomic['n_fit_scored_both_rounds'] = len(fit_both)
    for row in rows:
        if row['method'] in (*rules, *rules_both):
            for metric in (*METRICS, 'coverage', 'status'):
                atomic[f'{row["method"]}_{metric}'] = row[metric]
    for name, fit in fitted.items():
        atomic[f'{name}_parameters'] = json.dumps(asdict(fit), sort_keys=True)
    for name, fit in fitted_both.items():
        atomic[f'{BOTH_ROUNDS}{name}_parameters'] = json.dumps(asdict(fit), sort_keys=True)
    audit = {**base, 'config': asdict(config), 'fit_ids': fitting, 'eval_ids': evaluation,
             'fit_scored_ids': fit_scored, 'matched_ids': matched, 'fitting_accuracy': fitting_accuracy,
             'selection_reasons': dict(Counter(by_id[q]['post_debate']['selection_reason'] for q in evaluation)),
             'excluded': {q: {'shared_missing': [m for m, v in zip(models, by_id[q]['post_debate']['shared']) if v is None],
                              'initial_own_missing': [m for m, v in zip(models, by_id[q]['initial']['own']) if v is None],
                              'post_own_missing': [m for m, v in zip(models, by_id[q]['post_debate']['own']) if v is None],
                              'errors': by_id[q]['errors']} for q in evaluation if q not in matched_set},
             'sequence_interpretation': 'normalized over valid labels' if label_space(task) or cell.initial[models[0]][cell.ids[0]].get('options') else 'conditional on saved candidate set; no probability for unlisted answers',
             'reference': 'highest initial solo accuracy on fitting questions; order breaks ties',
             'pool_accuracy_all': vote_accuracy, 'numerical_precision': 12}
    return {'atomic': atomic, 'metrics': rows, 'predictions': predictions,
            'fits': {**{name: asdict(fit) for name, fit in fitted.items()},
                     **{BOTH_ROUNDS + name: asdict(fit) for name, fit in fitted_both.items()}}, 'audit': audit}
