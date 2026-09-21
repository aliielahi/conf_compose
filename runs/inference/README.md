# inference store

The only place that loads a model. Everything under `results/inferences/<task>/<model>--<decoding>--<digest>/`
is generated once and reused by every experiment; a cell is regenerated only if its digest is absent.

```bash
# greedy answers for the whole grid
python runs/inference/run_inference.py --tasks csqa boolq --models vllm/q3-4bi vllm/l31-8bi --dry-run

# five independent sampled voters per model
python runs/inference/run_inference.py --tasks csqa boolq --models vllm/q3-4bi \
  --answer-temperature 0.7 --voters 5 --no-verbalized
```

The digest covers task, model, split sizes, max tokens, answer temperature, voter index, and the consistency
sample count and temperature. Two cells with the same digest are interchangeable; two with different digests
are not, which is why the decoding tag is in the directory name.
