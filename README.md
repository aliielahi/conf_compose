# conf_compose

## zeroshot baselines and confidence

Run the whole grid (models x tasks from `src/conf_compose/constants.json`) detached: `bash runs/sweeps/sweep01-zs_baseline.sh` — add `--dry-run` first to see what is pending.

Each model loads once, answers every task greedily, and gets all confidence estimators (sequence probability, P(True) verification, verbalized, sampling consistency); per-run logs land in `logs/sweep01/`, per-run records and `report.json` in `results/sweep01/<task>/<model>_val<n>_test<n>/`, and finished runs are skipped on a rerun.

Summarize with `python runs/zeroshot_baseline/summarize.py` (writes `results/sweep01/summary.md` and `summary.csv`); `python runs/zeroshot_baseline/rereport.py --sweep sweep01` recomputes reports from saved records without a GPU.
