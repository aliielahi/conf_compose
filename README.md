# conf_compose

Confidence composition for multi-agent language model systems: estimate whether a system's answer is right by
pooling many agents' confidence signals.

## layout

- `src/conf_compose/` — all logic. Tasks, LLM backends, confidence estimators, metrics, pipelines, composition.
- `runs/inference/` — the only place that loads a model.
- `runs/experiment*/` — one experiment each, consuming the inference store and writing to `results/<same name>/`.
- `tests/` — regression tests over `src`.

## inference store

Every model output lives once in `results/inferences/<task>/<model>--<decoding>--<digest>/`. The digest covers
task, model, split sizes, max tokens, answer temperature, voter index, consistency samples and which estimators
ran, so two cells with the same digest are interchangeable and two with different digests are never confused.

```bash
# see what a request would generate
python runs/inference/run_inference.py --tasks gsm8k csqa --models vllm/q3-4bi --dry-run

# greedy answers over the full test split, every confidence signal
python runs/inference/run_inference.py --tasks gsm8k --models vllm/q3-4bi vllm/l31-8bi \
  --n-val 0 --n-test 0 --all-signals

# five independent sampled voters per model, for the voting experiment
python runs/inference/run_inference.py --tasks csqa boolq --models vllm/q3-4bi \
  --answer-temperature 0.7 --voters 5 --no-verbalized
```

`--n-test 0` means the whole split unsampled; omitting it uses the per-task default in `constants.json`.

## experiments

Each declares the inferences it needs, generates only what is missing, then analyses. `--local-only` refuses to
load a model and names what the store still owes.

| experiment | question |
|---|---|
| `experiment01-zeroshot_estimators` | which confidence estimator ranks one model's own answers best |
| `experiment02-voting_composition` | does pooling across independent voters beat one voter at matched budget |
| `experiment03-debate_composition` | does interaction change how useful the evidence is |

```bash
bash runs/experiment02-voting_composition/run.sh          # detached; writes logs/<name>/DONE when finished
cat results/experiment02-voting_composition/summary.txt
```

Experiment 02 needs one extra GPU pass before its target-verification arms have data:
`python runs/experiment02-voting_composition/score_targets.py --task csqa`.

## tests

```bash
pytest tests/ -q
```
