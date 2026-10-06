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

## Atomic consistency CSV

For member baselines and the three pooling rules in one comma-separated row per dataset/model group:

```bash
python runs/experiment02-voting_composition/run.py \
  --tasks csqa boolq gsm8k truthfulqa gpqa \
  --models vllm/q3-8bi vllm/l31-8bi vllm/g2-9i \
  --targets majority --voters 1 --sizes 3 --samples 5 --families cons \
  --n-val none --n-test all --all-signals --holdout 0.3 --local-only \
  --atomic-csv results/voting_atomic/q3-8bi_l31-8bi_g2-9i.csv
```

This mode is always offline and writes only the requested CSV (replacing it if it exists).
It leaves inference files unchanged and does not write the normal metrics/prediction reports.
List columns are JSON arrays inside properly quoted CSV fields. All member lists follow `models`
order; the three `*_ece_auarc` columns each contain `[ECE, AUARC]`. Values are fractions, not percentages.

The first columns are `task,n_models,models,samples_per_model,single_accuracy,single_ece,single_auarc,`
`voting_accuracy,mean_ece_auarc,logodds_sum_ece_auarc,logodds_mean_ece_auarc`.
Additional columns provide evaluation count, answer/confidence coverage, fraction with every model's
confidence available, mean source count on scored votes, tie rate, AUROC/Brier/NLL, scoring settings,
holdout count, evaluation-ID hash, inference directory names and composition-code hash.

Each member is evaluated on **its own answer**, and the pool on **that exact group's majority answer**,
using the same question IDs. Accuracy includes every evaluation question, counting missing answers as
incorrect. Confidence metrics use available scores; missing/all-invalid samples can reduce coverage.
Available models are pooled when some confidence sources are missing, so check `voting_full_source_coverage`.
`vote_tie_rate` is the fraction of all evaluation questions with a tied highest vote count. Ties choose
the first tied proposal in CLI model order (for two answered voters, this always selects the first model).

Consistency uses five samples and add-half smoothing `(support + 0.5)/(valid_samples + 1)`.
All reported metrics are raw: no calibration, reference selection or bootstrap is performed in this mode.
`--n-boot`, `--cap` and `--out-dir` do not apply. Undefined metrics (such as AUROC on one-class data)
are JSON `null` in list fields, or empty scalar fields. ECE uses ten equal-width bins.

The default `--holdout 0.3` preserves the earlier evaluation split; `--holdout 0` uses the full saved
test intersection when no validation split exists. The holdout is not used to fit anything in this mode.
Comparing groups requires checking that `eval_ids_sha256` matches, especially if inference rows are missing.

To enumerate all groups, list the available models and request `--sizes 2 3 4 5`. Atomic CSV mode gives
**each subset its own majority**, rather than the legacy report's majority across all loaded voters.
With seven models this yields 112 groups per dataset. To run one chosen group, list exactly its models
and set `--sizes` to that group's size.


## Atomic tables with learned pooling

Use `atomic.py` for all six methods in one row per dataset and exact model group:
`mean`, `logodds_sum`, `logodds_mean`, `shared_rho`, `shared_scale`, and `kahn`.
The numerical methods live in `src/conf_compose/composition/pooling/`.
The older `run.py --atomic-csv` path above still reports its original three fixed methods.

`shared_rho` centers stream logits separately within correct and incorrect majority-answer groups,
estimates their pooled within-class correlation, averages the pairwise correlations, clips the result
to [0, 1], and applies `1 / (1 + (S - 1) * rho)` to the summed logits. This is a restricted
exchangeability-based discount, not a full Bayesian dependence model.
`shared_scale` minimizes binary NLL over one positive multiplier of the summed logits, bounded
between 0.001 and 100. Neither method learns an intercept. Both preserve log-odds ranking at a fixed
panel size. Ranking metrics use logits to avoid artificial ties from sigmoid saturation.

By default, a deterministic hash of task, question ID, and `--fit-seed` reserves approximately
`--fit-fraction 0.3` of the saved test questions for fitting. Every method and member baseline is
then evaluated on the remaining questions. These metrics therefore use a different evaluation
set from earlier full-test tables. The fitting labels always describe the same majority answer
that every stream scores. `--fit-split validation` instead reads `validation.jsonl` from the same
selected inference cells and evaluates on all saved test questions; overlapping IDs are rejected.

```bash
python runs/experiment02-voting_composition/atomic.py \
  --tasks csqa boolq gsm8k truthfulqa gpqa \
  --group vllm/q3-8bi vllm/l31-8bi vllm/g2-9i \
  --samples 5 --voter 0 --match _cs7s --estimator cons \
  --fit-fraction 0.3 --fit-seed 0 \
  --out results/voting_atomic/learned_cons_kahn.csv

python runs/experiment02-voting_composition/atomic.py \
  --tasks csqa boolq gsm8k truthfulqa gpqa \
  --group vllm/q3-8bi vllm/l31-8bi vllm/g2-9i \
  --samples 5 --voter 0 --match _cs7s --estimator seq \
  --context direct --seq-score norm_sum \
  --fit-fraction 0.3 --fit-seed 0 \
  --out results/voting_atomic/learned_seq_kahn.csv
```

List more models and add `--sizes 2 3 4 5 6` to enumerate subsets, or omit `--sizes` to evaluate
exactly the listed group. Majority ties follow model order. This runner is offline and never
updates inference files. CSV rows append; use a new output path for a new experiment.
Old CSV headers are rejected rather than mixed with the new schema.

The table includes per-model accuracy and confidence metrics, majority accuracy, all six pools'
ECE/AUARC/AUROC/Brier/NLL and coverage, fitting counts, ID hashes, scales, raw/clipped rho,
optimizer bounds and fit statuses. Pools require a score from every member. Undefined learned
fits are left blank with a reason: shared rho requires at least two correct and two incorrect
fitting examples and nonzero within-class variance in every stream; shared scale requires both
classes and a nonzero summed logit. Boundary solutions are identified explicitly.


### Regularized Kahn aggregation

`kahn` implements the binary Gaussian log-odds aggregation model in Kahn (UAI 2004),
with a regularized estimate of its shared within-class covariance. For stream logits `z`,
fit the correct and incorrect class means `mu1` and `mu0`, then compute:

- `w = solve(Sigma_regularized, mu1 - mu0)`
- `b = logit(prior) - dot(w, (mu1 + mu0) / 2)`
- `confidence = sigmoid(b + dot(w, z))`

The prior is the fitting-set majority-answer correctness rate with an add-half pseudocount.
The covariance uses class-centered residuals and `n_fit - 2` degrees of freedom; it does not
include the between-class mean difference. Automatic oracle approximating shrinkage (OAS)
pulls this covariance toward a scaled identity, followed by a small positive diagonal ridge.
OAS uses the full finite-dimension formula in Chen et al., equation (23), with `n_fit - 2`
as the effective sample size. This regularization is an extension of Kahn's original rule.
It requires no search over the evaluation data and no additional dependencies.

Weights may be negative and need not sum to one. Unlike shared scale, Kahn can change ranking.
At least two correct and two incorrect fitting examples are required; otherwise its metrics
are unavailable with `kahn_status=insufficient_classes`. All constant inputs produce the
smoothed prior, and duplicate streams remain numerically stable under regularization.
The `kahn_parameters` CSV field records means, prior, weights, intercept, both covariance
matrices, shrinkage and ridge. `kahn_scale` is empty because this method has a vector of weights.
The Python API accepts `fit_kahn(scores, labels, shrinkage=..., ridge=...)` for later experiments;
`shrinkage=None` is the automatic default and numeric shrinkage must lie in [0, 1].

Use the same atomic commands above with fresh `--out` paths: the extra Kahn columns cannot be
appended to earlier five-method CSV files. This is a regularized literature baseline, not a
claim of guaranteed superiority or an unmodified reproduction of the paper's experiments.

Sources: https://arxiv.org/abs/1207.4144 and https://arxiv.org/abs/0907.4698.


## Learned pooling on the judge panels

`atomic.py` now reports eight default rules: `mean`, `logodds_sum`,
`logodds_mean`, `shared_rho`, `shared_scale`, `kahn`, `blp`, and
`logistic_pool`. The old `run.py --atomic-csv` path is unchanged.

- `blp` is the weighted beta-transformed linear pool: Beta CDF of a convex
  weighted arithmetic pool. It fits simplex weights and two positive shape
  parameters by fitting-set NLL. Three fixed optimizer starts are compared on
  fitting NLL only; shape bounds are [0.05, 100]. The pooled probability is
  clipped to [0.001, 0.999] for stable fitting and prediction.
- `logistic_pool` fits an intercept and separate signed logit weights with
  L2 regularization, using fitting-set means and standard deviations. Its
  objective is summed NLL plus `logistic_l2 / 2 * ||standardized_weights||²`;
  the intercept is unpenalized. Default `--logistic-l2 1` is fixed, not chosen
  using evaluation results. Signed weights can downweight redundant evidence
  or reverse a conditionally misleading stream; they are not estimates of
  pairwise correlation.
- `--ablations` additionally reports `blp_equal` (only two fitted beta shapes)
  and `kahn_diagonal` (the same Kahn means, prior, shrinkage and diagonal
  variances, with off-diagonal covariance removed).

Every learned fit uses only the fitting questions. Fewer than two examples
of either class, or optimizer failure, produces an explicit unavailable result,
not a silently substituted baseline. CSVs retain fit statuses and parameters.

The following command runs the exact 15 ordered judge panels, for both
consistency and direct candidate sequence probability, on five datasets:

```bash
python -m pip install "scipy>=1.12"
python runs/experiment02-voting_composition/sweep_learned.py \
  --tasks csqa boolq gsm8k truthfulqa gpqa \
  --estimators cons seq --samples 5 --match _cs7s \
  --fit-fraction 0.3 --fit-seed 0 --ablations \
  --out-dir results/voting_atomic/learned_panels_v1
```

It writes `cons.csv` and `seq.csv`, 75 rows each. Each row is one exact
panel/task/estimator, with individual-model metrics, majority accuracy, and
ECE/AUARC/AUROC/Brier/NLL for each rule. Input hashes, code hash, split ID
hashes, fit counts, learned weights and optimizer diagnostics are recorded.
No new inference runs. The judge and this sweep share `composition_panels`
in `constants.json`, including panel order, which matters for tied votes.

`--dry-run` prints the panels without opening results or writing output.
`--panel-sizes 2 3` restricts the existing panel list; it does not generate new
combinations. Existing output files are refused: use a fresh output directory.
Rows are saved incrementally, so completed rows survive an interrupted run.

By default, 30% of saved test question IDs are reserved for fitting and excluded
from evaluation, using a stable hash split. These are held-out comparisons;
they are not full-test metrics. `--fit-split validation` instead requires
separate, non-overlapping validation files. Comparisons with judge results
must use these same evaluation IDs. Sequence scores on open-answer tasks
remain conditional on the saved candidate set; this runner does not change
that estimator or claim those scores account for all unobserved answers.

Do not pick regularization values or method variants by their test results.
Use fitting-only cross-validation if tuning `--logistic-l2` later. Compare
`blp` with `blp_equal` to isolate weighting, and `kahn` with `kahn_diagonal`
to isolate covariance terms. BLP's beta transform is monotone: for fixed
weights it changes calibration, not ranking; learned weights can change both.

BLP ranking metrics use the weighted arithmetic score before the monotone
beta transform; probability metrics use the transformed probability. This
avoids artificial ranking changes from CDF rounding or probability clipping.
In particular, equal-weight BLP preserves the arithmetic baseline's AUROC
and AUARC exactly on the same evaluated questions.

BLP reference: Ranjan and Gneiting (2010), Combining Probability Forecasts,
https://academic.oup.com/jrsssb/article/72/1/71/7076442.
