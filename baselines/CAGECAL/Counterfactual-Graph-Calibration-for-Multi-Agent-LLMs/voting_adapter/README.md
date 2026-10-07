# CAGE-Cal on saved voting panels

This adapter adds **CAGE-Cal (IID adaptation)** to the existing voting comparison.
It uses the upstream `HyperHybridGNN`, graph builder and Pearson-correlation rule.
It does not run your answering LMs, edit their inferences, or modify the parent
project. All caches, checkpoints, predictions and table copies stay in this
CAGE-Cal repository.

## On the GPU container

Copy this `voting_adapter/` directory into the same CAGE-Cal repository on the
GPU, alongside the existing `scripts/` and `cage_cal/` directories. The parent
project needs its saved inferences, the `learned_panels_v1` CSVs and the current
`paper_results/results/voting_protocol/{audit.json,atomic.csv}`. Keep their
relative directory layout. No Git commands are required by this adapter.

```bash
cd /workspace/baselines/CAGECAL/Counterfactual-Graph-Calibration-for-Multi-Agent-LLMs
python -B voting_adapter/setup_gpu.py
source .venv-voting/bin/activate
```

The setup creates `.venv-voting` with access to the container's existing PyTorch.
It installs the smaller graph/embedding/calibration dependencies into that venv
using `--no-deps` calls and checks their dependency metadata. It refuses to
install or upgrade torch, torchvision, torchaudio, Triton or CUDA packages.
This avoids replacing the working CUDA build. The GPU setup has not been run on
this Mac. If it reports an environment error, stop there and keep the traceback.

First check the actual graph model and the data without downloading an embedding model:

```bash
python -B -m voting_adapter.run --smoke --device cuda:0
python -B -m unittest voting_adapter.test_adapter -v
python -B -m voting_adapter.run --audit-only
```

Then run a short end-to-end pilot, using one existing three-model group:

```bash
python -B -m voting_adapter.run \
  --tasks csqa boolq \
  --panel q3-4bi l31-8bi g3-12i \
  --seeds 1 --epochs 2 --device cuda:0
```

This tests the pipeline; two epochs and one seed are **not the final baseline**.
For the full comparison on the existing 75 task/panel combinations:

```bash
python -B -m voting_adapter.run --device cuda:0
```

Defaults: 10 seeds, 15 maximum epochs, early stopping after five unimproved
validation epochs, batch size 128, 20 nearest training questions for W. Use
`--tasks csqa boolq` to restrict datasets or `--panel ...` to select exactly one
ordered existing group. `--project /path/to/conf_compose` overrides automatic
parent-project discovery. New datasets/panels must first exist in the parent
project's paper-table audit; this adapter deliberately does not invent splits.

## What is comparable, and what is adapted

- **Same answers:** the saved voter-0 answer from each model; task equivalence
  and first-model tie-breaking match the existing majority vote. The five
  consistency samples are not extra CAGE graph nodes or extra voting agents.
- **Same evaluation:** source SHA-256 hashes and fit/evaluation/matched-ID hashes
  are verified. Every exported row uses exactly the questions in its original
  paper-table cell. Missing graph inputs may exclude examples outside that mask;
  a missing input inside the mask aborts instead of changing the comparison.
- **CAGE features:** answer-string embeddings, the mean token log-probability of
  each saved response, vote ranks and local correctness correlations. Full
  reasoning text is not embedded by the upstream method. For multiple-choice
  questions the embeddings describe the saved labels, as in the upstream code;
  label-to-option mappings are checked across models before constructing a vote.
- **IID topology:** both towers receive the same independent panel and the same
  W; communication adjacency is zero. Pairwise dependency edges and hyperedges
  still exist. All agents have the voter role. This is a voting adaptation, not
  a reproduction of the paper's full interacting-topology experiment.
- **Supervision:** a single GNN is trained jointly across the requested tasks
  and panels. The existing outer fit/evaluation split is preserved. Within the
  outer fitting questions, 20% are reserved for validation by a deterministic
  label-independent ordering. Every panel uses the same question partition, so
  overlapping panels cannot put a question in both training and evaluation.
  The pooling methods were fit per panel; this difference must be disclosed.
- **No label leakage into W/PCA:** W is estimated from nearest *training*
  questions, excluding the query itself. It uses one saved execution per model,
  rather than upstream's three rollouts. Answer PCA is fitted on training answer
  strings only and padded to 16 dimensions when there are too few distinct labels.
- **Training objective:** upstream BCE with 0.05 label smoothing plus 0.4 Brier
  loss, AdamW at 0.002, weight decay 0.0003, cosine schedule, hidden size 64,
  four attention heads, dropout 0.3. Selection uses validation AUROC minus half
  ECE averaged over tasks with both classes; if none qualify, negative Brier is
  the documented fallback. Model weights are averaged through predicted
  probabilities, not parameter averaging.
- **Calibration:** export both the raw seed ensemble and upstream-style Beta +
  Platt-binning calibration. These fits use the same internal validation set as
  early stopping, as upstream does. Calibration requires at least 50 distinct
  validation questions and both classes. Otherwise that task uses identity;
  component failures and fallbacks are recorded. With the current split,
  TruthfulQA and GPQA have fewer than 50 validation questions, so their BetaSB
  row falls back to raw. Do not present that fallback as a fitted calibrator.
- **Cons/seq tables:** CAGE produces one confidence per fixed vote, independently
  of those estimator families. That same prediction appears in both table sets,
  evaluated on each set's original matching mask; it is not retrained separately
  or fed the pooled consistency/sequence scores.

## Outputs and resuming

The runner prints `results/voting_adapter/<digest>/`. The digest covers inputs,
source code, panel/task configuration and question splits. Completed seed
checkpoints are reused when repeating the identical command. An interrupted
seed restarts from the beginning of that seed. Changed dependencies require a
fresh `--output-root` within this baseline repository.

- `manifest.json`, `runtime.json`: source hashes, split IDs, constants, adaptations,
  package versions and runtime device.
- `features.jsonl`: per-example answers, log-probs, W and training-neighbor IDs.
- `seed_*.pt`, `seed_*_training.json`, `seed_*_predictions.npz`: checkpoints,
  validation-selection history and seed-level predictions.
- `predictions.jsonl`: fixed vote, correctness, raw/BetaSB probability, all seeds.
- `calibration.json`, `calibrators.pkl`: component fits/fallbacks and saved objects.
- `paper_tables/cagecal_metrics.csv`: exact per-task, per-panel ECE, AUARC, AUROC,
  Brier, NLL, accuracy and coverage for both CAGE rows.
- `paper_tables/atomic.csv`: original table rows plus the CAGE rows.
- `paper_tables/{cons,seq}/*.tex` and `*.txt`: augmented table copies in the
  existing paper format, including individual panel sizes and coverage.

`calibrators.pkl` is your own local artifact; only load trusted pickle files.
The parent `paper_results` directory is never rewritten. Copy the generated
LaTeX tables to Overleaf after checking the atomic rows. No significance claim
is inferred from these point estimates.

To rebuild table copies, repeat the original task/panel/training arguments and
add `--report-only results/voting_adapter/<digest>`. This verifies the manifest
and does not train or download models.

## Local checks

`python -B -m unittest voting_adapter.test_adapter -v` tests task-equivalent votes,
ordered ties, question-grouped splits, PCA/W leakage protection, degenerate W,
output containment and exact-mask table export. The graph test runs if PyTorch
and PyG are installed; otherwise it is explicitly skipped. `--audit-only` uses
only the standard library and the parent project's task parsers, performs no
inference, and writes no output files. Training and MiniLM downloads are left to
the GPU environment.
