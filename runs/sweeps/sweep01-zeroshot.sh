#!/usr/bin/env bash
# Sweep 01: zero-shot baselines for every model x task in src/constants.json, detached with nohup.
# Usage: bash runs/sweeps/sweep01-zeroshot.sh [--parallel N] [-- extra eval_zero_shot.py flags]
set -euo pipefail
cd "$(dirname "$0")/../.."

SWEEP=sweep01
mkdir -p "logs/$SWEEP" "results/$SWEEP"
nohup python runs/zeroshot_baseline/run_baselines.py --sweep "$SWEEP" "$@" > "logs/$SWEEP/sweep.log" 2>&1 &
echo "$!" > "logs/$SWEEP/sweep.pid"
echo "started $SWEEP (pid $!)"
echo "progress: tail -f logs/$SWEEP/sweep.log"
echo "summary:  python runs/zeroshot_baseline/summarize.py --sweep $SWEEP"
