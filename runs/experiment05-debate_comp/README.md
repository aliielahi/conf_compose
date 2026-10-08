# Experiment 05: post-debate confidence composition

This runner reads experiment03's saved conversations and scores. It generates no
answers and does not change the inference store or paper reports. The group is
exactly the group that debated; models are never mixed across debate cells.

```bash
python runs/experiment05-debate_comp/run.py --tasks gpqa --require-complete
```

When more datasets arrive, pass their names, or omit `--tasks` to discover all
available datasets. Incomplete cells are listed and skipped by default;
`--require-complete` makes an incomplete selected cell an error. `--dry-run`
checks source records without fitting or writing results. `--match` selects a
specific saved run if the store contains multiple versions of a group.

Defaults: round 1, five consistency samples, both `cons` and `seq`, direct-context
candidate log-probability sums, and all ten pooling methods. To evaluate later
rounds, use `--rounds 1 2`. Each requested round needs its own
`candidates_through_round_<round>` files when sequence scoring is enabled.
`--estimators cons` works without candidate sidecars. `--context reasoned` and
`--seq-score norm_mean` select the other saved sequence variants.

## Evaluation

- Select the post-round majority answer once per question. Vote ties use the sum
  of supporters' smoothed own-answer consistency, then the same seeded tie rule
  as experiment02. Selection is independent of the estimator and pooling rule.
- Consistency measures support for that shared target using the first five saved
  samples and add-half smoothing. Invalid samples are omitted from the valid
  denominator; their counts are saved. Zero support is a score, not missing data.
- Sequence scoring reads the requested round's candidate sidecar, never the
  initial answer's self-score. Direct scores condition on the saved debate
  conversation; reasoned scores also condition on the generated reasoning.
  Closed-label tasks require every valid label. For open answers, normalization
  is conditional on the saved candidate set and does not estimate all-wrong mass.
- Reserve questions for fitting with experiment02's identical 30/70 hash rule
  and seed 0, before filtering scores. Fit each group, round and estimator
  independently. Evaluation labels never enter a fit or reference selection.
- Reuse the existing arithmetic mean, log-odds sum/mean, shared rho/scale, full
  Kahn, weighted BLP and regularized logistic pooling, plus equal-weight BLP and
  diagonal Kahn. `--no-ablations` omits the latter two. Unavailable fits retain
  their status and missing metrics; no replacement method is silently used.
- Report initial solo and post-debate solo models as well as the pools. The
  reference is the initial solo model with highest fitting accuracy, with group
  order breaking ties. Every metric uses that same reference. Deltas are
  computed within the group, not after averaging groups.
- Confidence comparisons share the intersection of available initial solo,
  post-debate solo and shared-target scores within each group/estimator.
  Coverage and excluded IDs/reasons are explicit. `accuracy_all` and `vote_acc`
  use every evaluation question, counting absent answers as incorrect; the
  `accuracy` column in metrics.csv is for the common scored subset.

## Outputs

Everything is under `results/debate_composition/run_<hash>/`. Input hashes,
settings, source code and numerical-library versions define the run identity.
A repeat verifies and reuses completed outputs. A changed input or setting
creates a new run. Files are published only after the run finishes; interrupted
work has a `.partial` directory, never a completed manifest.

- `cons.csv`, `seq.csv`: one wide atomic row per dataset, group and round,
  including member metrics, vote accuracy, every pooling metric and fitted values.
- `metrics.csv`: long rows for every initial member, post-debate member, pool and
  solo reference, with accuracy/ECE/AUARC/AUROC/Brier/NLL and reference deltas.
- `<task>/<cell>/round_<r>/<estimator>/predictions.jsonl`: every question's split,
  selected answer, initial/post answers, member scores, pooled probabilities,
  ranking scores, correctness labels and evaluation-mask membership.
- `audit.json` and `fits.json` alongside predictions: exact fitting/evaluation
  IDs, exclusions, reference choice, fit status and all fitted parameters.
- `manifest.json`: provenance, artifact hashes, configuration and skipped cells.

Each cell is loaded separately to keep memory bounded as datasets accumulate.
The loader verifies the original round-0 file hashes even after GPU paths are
relocated, and rejects duplicate IDs, mismatched labels/options, wrong group or
round metadata, and partial candidate files. Recorded generation errors remain
in the accuracy denominator and are excluded from confidence comparisons.

```bash
python -m unittest discover -s runs/experiment05-debate_comp -p 'test_*.py' -v
python runs/experiment05-debate_comp/check.py --run results/debate_composition/run_<hash>
```

The checker replays saved pooling parameters and recomputes reported metrics. It
also verifies input/output hashes, split separation and identical selected
answers across estimators. No inference, refitting, or paper-table generation.
