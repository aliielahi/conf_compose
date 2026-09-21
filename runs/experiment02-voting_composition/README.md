# experiment02 — voting composition

**Question.** For one fixed answer, does pooling confidence across independent voters beat the best single
voter, and does spending a fixed generation budget on *distinct models* beat spending it on more runs of one
model?

**Datasets.** CSQA and BoolQ. **Models.** q3-4bi, l31-8bi, g3-12i, phi4mii. **Voters.** 5 sampled answers per
model at T=0.7, each with its own 5 consistency samples.

**Varied.** Panel size 2–5; arm (distinct models / same model / one run's samples split); samples per stream
1–5; estimator family (consistency, verification, target verification, and combinations); target (the anchor's
answer, the panel's majority answer); pooling rule (arithmetic mean, log-odds sum, log-odds mean).

```bash
bash runs/experiment02-voting_composition/run.sh              # fills the store if needed, then analyses
python runs/experiment02-voting_composition/run.py --local-only   # refuses to load a model; names what is missing
```

Inferences come from `results/inferences/`; nothing here generates a model output except through
`ensure_inference`. Outputs land in `results/experiment02-voting_composition/` with `summary.txt` at its root.
