# conf_compose

## zeroshot baselines and confidence

Run the whole grid (models x tasks from `src/conf_compose/constants.json`) detached: `bash runs/sweeps/sweep01-zs_baseline.sh` — add `--dry-run` first to see what is pending.

Each model loads once, answers every task greedily, and gets all confidence estimators (sequence probability, P(True) verification, verbalized, sampling consistency); per-run logs land in `logs/sweep01/`, per-run records and `report.json` in `results/sweep01/<task>/<model>_val<n>_test<n>/`, and finished runs are skipped on a rerun.

Summarize with `python runs/zeroshot_baseline/summarize.py` (writes `results/sweep01/summary.md` and `summary.csv`); `python runs/zeroshot_baseline/rereport.py --sweep sweep01` recomputes reports from saved records without a GPU.

## voting panels

Sweep the offline voting grid on the saved zero-shot records (no GPU): `bash runs/sweeps/sweep03-voting_panels.sh` — set `SOURCE=results/sweepNN` to point at a different generation sweep.

Each cell fixes one target answer (the anchor's, and the ensemble majority) and varies panel size 2-5, model set (heterogeneous over every model subset against homogeneous sub-streams of one model), samples per stream 1-5, and estimator family (consistency, verification, verbalized); logs land in `logs/sweep03/`, metrics, per-example predictions and a manifest in `results/composition/<task>/panels_<target>_<hash>/`.

Summarize with `python runs/composition/summarize_panels.py --metric auroc --rule mean` for the panel-size curve, the matched sampled-generation comparison, and the family and rule cuts.

## independent voters (GPU)

`VOTERS=5 bash runs/sweeps/sweep04-voters.sh` generates, per model and task, VOTERS separate answers at T=0.7, each with its own consistency samples and its own sampling identity (`_rep<i>` directories); `DRY_RUN=1` lists pending work first and `--skip-existing` makes it resumable.

Then `bash runs/sweeps/sweep04-score_targets.sh` has every model verify the panel's anchor answer and its majority-vote answer, writing `results/sweep04/<task>/target_verification_<model>_<split>.json`.

Feed both into the offline panels with `--validation 'results/sweep04/<task>/*/validation.jsonl' --test '...' --target-scores 'results/sweep04/<task>/target_verification_*.json'`.
