import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from conf_compose.data import Example, get_task


class IncompleteCell(ValueError):
    pass


@dataclass
class DebateRecords:
    directory: Path
    task: str
    models: tuple[str, ...]
    round: int
    ids: list[str]
    initial: dict
    current: dict
    settings: dict
    sources: dict[str, str]


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read_records(path, keys=('id',)):
    if not path.is_file():
        raise IncompleteCell(f'missing {path}')
    rows = {}
    with path.open() as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise IncompleteCell(f'invalid JSON at {path}:{number}') from error
            key = tuple(row[field] for field in keys)
            key = key[0] if len(key) == 1 else key
            if key in rows:
                raise ValueError(f'duplicate record {key} in {path}')
            rows[key] = row
    return rows


def load_cell(directory, round_index, inference_store, need_sequence=True):
    directory, inference_store = Path(directory), Path(inference_store)
    path = directory / 'settings.json'
    payload = json.loads(path.read_text())
    settings = payload['settings']
    models = tuple(settings['group'])
    if round_index < 1 or len(models) < 2 or len(set(models)) != len(models):
        raise ValueError(f'invalid debate round or model group: {directory}')
    task = get_task(settings['task'])
    sources = {str(path): file_digest(path)}

    def read(path, keys=('id',)):
        before = file_digest(path) if path.is_file() else None
        rows = read_records(path, keys)
        if file_digest(path) != before:
            raise IncompleteCell(f'input changed while reading: {path}')
        sources[str(path)] = before
        return rows

    initial = {}
    for model in models:
        source = payload['round0'][model]
        saved = Path(source['path'])
        path = inference_store / task.name / saved.parent.name / saved.name
        initial[model] = read(path)
        if sources[str(path)] != source['sha256']:
            raise ValueError(f'round-0 hash mismatch: {path}')
    ids = [key for key in initial[models[0]] if all(key in initial[model] for model in models)]
    if settings.get('limit'):
        ids = ids[:settings['limit']]
    if not ids:
        raise ValueError(f'no shared questions: {directory}')
    expected = set(ids)
    current = {}
    for agent, model in enumerate(models):
        path = directory / f'round_{round_index}' / f'{model}.jsonl'
        rows = read(path)
        if set(rows) != expected:
            raise IncompleteCell(f'round {round_index}, {model}: expected {len(expected)} question IDs, found {len(rows)}')
        for question in ids:
            row = rows[question]
            if (row.get('model'), row.get('round'), row.get('agent'), row.get('group')) != (model, round_index, agent, list(models)):
                raise ValueError(f'wrong agent/round/group metadata: {path}, {question}')
            if not row.get('messages'):
                raise ValueError(f'missing debate conversation: {path}, {question}')
            original = initial[models[0]][question]
            for record in (row, initial[model][question]):
                for field in ('gold', 'question', 'options', 'choices'):
                    if record.get(field) != original.get(field):
                        raise ValueError(f'{field} mismatch: {path}, {question}')
                example = Example(question, record['question'], record['gold'])
                if bool(record['correct']) != task.is_correct(record['prediction'], example):
                    raise ValueError(f'incorrect saved correctness label: {path}, {question}')
            if row.get('error') and row.get('prediction') is not None:
                raise ValueError(f'failed turn contains an answer: {path}, {question}')
        current[model] = rows
        if need_sequence:
            path = directory / f'candidates_through_round_{round_index}' / f'{model}.jsonl'
            scores = read(path, ('round', 'id'))
            wanted = {question for question in ids if not rows[question].get('error')}
            available = {question for (r, question) in scores if r == round_index}
            if available != wanted:
                raise IncompleteCell(f'candidate-score IDs incomplete for round {round_index}, {model}')
            if any(question not in expected or r < 0 or r > round_index for r, question in scores):
                raise ValueError(f'unexpected candidate-score round or question in {path}')
            for question in wanted:
                current[model][question] = {**rows[question], 'candidate_scores': scores[round_index, question]['candidate_scores']}
    for question in ids:
        pools = [current[model][question]['candidate_scores']['candidates'] for model in models
                 if need_sequence and not current[model][question].get('error')]
        if pools and any([c['answer'] for c in pool] != [c['answer'] for c in pools[0]] for pool in pools[1:]):
            raise ValueError(f'candidate sets differ across models: {directory}, {question}')
        for pool in pools:
            for candidate in pool:
                option = initial[models[0]][question].get('choices') or {}
                if candidate['answer'] in option and candidate.get('option') != option[candidate['answer']]:
                    raise ValueError(f'candidate label/text mapping mismatch: {directory}, {question}')
    return DebateRecords(directory, task.name, models, round_index, sorted(ids), initial, current, settings, sources)
