# Voting protocol paper results

The default version uses majority vote with consistency-confidence tie-breaking,
with five samples per model. Vote counts decide first; tied candidates are ordered
by the sum of their supporters' add-half-smoothed own-answer consistency scores.
Ties requiring a missing consistency score and remaining exact ties use a reproducible hash of the seed, task, question, model set,
and answer aliases. Both confidence estimators use the same selection rule.

## Run

From the project root, regenerate the tables from the new saved fits:

```bash
python paper_results/codes/voting_protocol/run.py --ablations --significance
python paper_results/codes/voting_protocol/plots/run.py
```

To refit from the original inference and splits into a fresh directory:

```bash
python paper_results/codes/voting_protocol/run.py --refit --ablations --significance \
  --pool-dir results/voting_atomic/learned_panels_confidence_tie_v3
```

No model inference is performed. `--refit` never overwrites existing pooling CSVs.
It preserves the original fitting/evaluation question IDs and input hashes, selects
new targets on each split, and fits all learned methods on fitting questions only.
Without `--refit`, fitted parameters are restored unchanged. Selection-rule mismatches
between requested tables and saved fits raise an error. `--tie-break first` remains
available for the original selection policy; it requires matching saved fits.

## Versions and inputs

- Original pooling fits remain in `results/voting_atomic/learned_panels_v1/`.
- New default pooling fits are in `results/voting_atomic/learned_panels_confidence_tie_v2/`.
- Original tables, plots, audit and manifest are preserved in
  `results/voting_protocol_archives/first_tie_fit_metric_v1_2026-10-07/`.
- Current paper outputs are in `paper_results/results/voting_protocol/`.
- Inference records are read from `results/inferences/`.
- Existing judge scores are read from `results/experiment04-judge_baseline/`.

The original archive used fitting-metric-specific references. The new default uses
one fitting-accuracy-selected reference per group. Differences between the old and
new reported deltas therefore include both selection and reference changes; use
absolute accuracy on identical questions to isolate the selection change.

## Reference and aggregation

Default `--reference fit_accuracy`: choose the highest-accuracy group member using
only the fitting questions, breaking ties by the recorded panel order. Use this
same model as reference for accuracy, ECE, AUARC, AUROC, Brier and NLL on held-out
questions. This avoids choosing the reference from test performance.

For every method and group, calculate `method metric - reference metric` first.
Only then average the deltas over groups within each dataset. Each of the 15 groups
has equal weight; the overall result is not an average of size-category averages.
Size tables use only groups of that size. Negative ECE/Brier/NLL and positive
accuracy/AUARC/AUROC deltas indicate improvement over the declared reference.

Other options remain available: `fit_metric` chooses each metric's reference on
fitting data; `eval_accuracy` chooses by evaluation accuracy; `metric_best` gives
an evaluation-set oracle envelope. The last two are descriptive, not deployable
reference-selection rules. Use a separate output directory for these comparisons.

Solo members answer independently, whereas all pooling and judge rows rate the
selected voting answer. Thus AUARC changes versus solo can partly reflect accuracy
changes. All non-solo methods share voting accuracy on the matched question set.

## Approximate judge reuse

The temporary default `--judge-policy approximate` reuses the existing verbalized
judge confidence even when the new tie-break changes the answer. These scores are
proxies, not valid newly elicited confidence in the changed target. Judge rows are
labeled `(approx.)` in tables and plots; `atomic.csv` records mismatch counts, and
`audit.json` records the exact affected question IDs and old/new answers.

All present methods and solo models use the same question mask within each
panel/estimator, including judge score availability. Cross-estimator masks can
differ; coverage is reported. Approximate judges are excluded from significance
tests. Judge inferences must be rerun before publication; switch to
`--judge-policy strict` once their targets match the selected answer.

The two judges are `g3-27i` and `l32-3bi`, each with reasoning-only and
reasoning-plus-confidence views. Original inference files and judge files are
never modified. A mismatched correctness label for an identical target remains
an error even in approximate mode.

## Outputs

Each `cons/` and `seq/` directory contains TXT and LaTeX versions of:

- `all_delta`, `all_absolute`: ECE/AUARC, as before.
- `all_full_delta`, `all_full_absolute`: Acc, ECE, AUARC, AUROC, Brier and NLL,
  with one accuracy subcolumn per dataset and method row.
- `all_accuracy_delta`, `all_accuracy_absolute`: a compact accuracy-only table.
- Equivalent tables prefixed `size_2` through `size_6`.
- Separate coverage and exploratory significance tables.

All metrics except NLL are displayed multiplied by 100; NLL is in nats. Full
six-metric tables are wide, so use a landscape page or resize them for Overleaf.
Fragments use `booktabs` and can be inserted with `\\input{...}`. Raw CSV values
are unscaled. AUROC is unavailable for one-class question sets, never invented.

`atomic.csv` contains each dataset/group/estimator/method's absolute metrics,
within-group deltas, reference model, coverage, selection rule and judge flags.
`audit.json` contains source hashes, fitting metrics, target changes, matched IDs,
all solo metrics, judge mismatches, selection fallback reasons and replay differences. `manifest.json` records
inputs, code hashes, aggregation choices and all table paths.

## Significance and plots

`--significance` computes exploratory one-sided sign tests on dataset-level mean
improvements and applies Holm correction jointly across methods, estimators and
ECE/AUARC. Approximate judges are excluded: with all pooling ablations there are
40 tests. Five datasets allow a minimum raw p-value of 1/32, insufficient to pass
this correction. Absence of significance does not establish equivalence.

Panels share models and questions; treating them as independent replications
would exaggerate the evidence. Existing panel-resampling intervals in plots are
descriptive. A future question-level paired bootstrap should resample question
IDs jointly across overlapping groups while keeping saved fits fixed.

`plots/run.py` regenerates all PNG/PDF figures, including per-dataset, per-estimator
heatmaps showing all 15 groups against all methods. Figures label judge reuse as
approximate. Main tables stay free of coverage superscripts and significance stars.
