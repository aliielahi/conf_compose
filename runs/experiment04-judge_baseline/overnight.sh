#!/usr/bin/env bash
# Overnight judge sweep: 15 panels x 5 datasets x 2 judges x (reasoning control + 2 confidence estimators).
# One judge per process, because two vLLM engines cannot share one GPU at 0.7 utilisation each.
# Datasets run cheapest first, so an interrupted night leaves whole datasets finished.
# Rerunning skips every cell already on disk, so this is the resume command too.
# Usage: bash runs/experiment04-judge_baseline/overnight.sh
#   TASKS="gpqa csqa"      only these datasets          JUDGES="vllm/g3-27i"   only this judge
#   SIZES="2 3"            only panels of these sizes   EXTRA="--no-control"   drop the control arm
#   EXTRA="--dry-run"      print the plan and exit
set -euo pipefail
cd "$(dirname "$0")/../.."

JUDGES=${JUDGES:-"vllm/g3-27i vllm/l32-3bi"}
TASKS=${TASKS:-"gpqa truthfulqa csqa gsm8k boolq"}
LOGS=${LOGS:-logs}
mkdir -p "$LOGS"

for JUDGE in $JUDGES; do
  TAG=$(echo "$JUDGE" | tr '/' '-')
  LOG="$LOGS/judge_sweep-$TAG-$(date +%Y%m%d_%H%M).log"
  echo "################ $JUDGE  ->  $LOG"
  python runs/experiment04-judge_baseline/sweep.py \
    --judges "$JUDGE" --tasks $TASKS \
    ${SIZES:+--panel-sizes $SIZES} ${EXTRA:-} 2>&1 | tee "$LOG"
done

echo "################ done; summary at results/experiment04-judge_baseline/summary.csv"
