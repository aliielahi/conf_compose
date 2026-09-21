# experiment01 — zero-shot confidence estimators

**Question.** For a single model answering greedily, which confidence estimator best separates its correct
answers from its wrong ones, and how well calibrated is each?

**Datasets.** All five tasks. **Models.** q3-4bi, l31-8bi, g3-12i, phi4mii. **Decoding.** Greedy answers, with
sequence probability, P(True) verification, verbalized confidence and sampling consistency attached.

```bash
bash runs/experiment01-zeroshot_estimators/run.sh
python runs/experiment01-zeroshot_estimators/run.py --local-only   # names any missing inference
```

Outputs land in `results/experiment01-zeroshot_estimators/reports.json`.
