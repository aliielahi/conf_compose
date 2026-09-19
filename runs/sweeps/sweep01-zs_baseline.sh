#!/usr/bin/env bash
# Zero-shot confidence baselines: the whole model x task grid from src/conf_compose/constants.json, under nohup.
# Usage: bash runs/sweeps/sweep01-zs_baseline.sh [--parallel N] [--dry-run] [-- extra eval_zero_shot.py flags]
set -euo pipefail
cd "$(dirname "$0")/../.."

SWEEP=sweep01
mkdir -p "logs/$SWEEP" "results/$SWEEP"
nohup python runs/zeroshot_baseline/run_baselines.py --sweep "$SWEEP" "$@" > "logs/$SWEEP/sweep.log" 2>&1 &
echo "$!" > "logs/$SWEEP/sweep.pid"
echo "started $SWEEP (pid $!)"
echo "progress: tail -f logs/$SWEEP/sweep.log"
echo "summary:  python runs/zeroshot_baseline/summarize.py --sweep $SWEEP"
