#!/usr/bin/env bash
# Overnight judge sweep: 15 panels x 5 datasets x 2 judges x (reasoning control + 2 confidence estimators).
# One judge per process, because two vLLM engines cannot share one GPU at 0.7 utilisation each.
# Datasets run cheapest first, so an interrupted night leaves whole datasets finished.
# Rerunning skips every cell already on disk, so this is the resume command too.
# Progress and crashes land in logs/current_run.log; each judge also gets its own full stdout log.
# Usage: bash runs/experiment04-judge_baseline/overnight.sh
#   TASKS="gpqa csqa"      only these datasets          JUDGES="vllm/g3-27i"   only this judge
#   SIZES="2 3"            only panels of these sizes   EXTRA="--no-control"   drop the control arm
#   EXTRA="--dry-run"      print the plan and exit
set -uo pipefail
cd "$(dirname "$0")/../.."

JUDGES=${JUDGES:-"vllm/g3-27i vllm/l32-3bi"}
TASKS=${TASKS:-"gpqa truthfulqa csqa gsm8k boolq"}
LOGS=${LOGS:-logs}
RUN_LOG=$LOGS/current_run.log
mkdir -p "$LOGS"

note() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $*" >> "$RUN_LOG"; }

note "$(printf '=%.0s' {1..78})"
note "SHELL    overnight.sh starting: judges=$JUDGES tasks=$TASKS"

STATUS=0
for JUDGE in $JUDGES; do
  TAG=$(echo "$JUDGE" | tr '/' '-')
  LOG="$LOGS/judge_sweep-$TAG-$(date +%Y%m%d_%H%M).log"
  echo "################ $JUDGE  ->  $LOG"
  note "SHELL    launching $JUDGE, stdout at $LOG"
  python runs/experiment04-judge_baseline/sweep.py \
    --judges "$JUDGE" --tasks $TASKS --log "$RUN_LOG" \
    ${SIZES:+--panel-sizes $SIZES} ${EXTRA:-} 2>&1 | tee "$LOG"
  CODE=${PIPESTATUS[0]}
  if [ "$CODE" -ne 0 ]; then
    STATUS=$CODE
    note "SHELL    $JUDGE EXITED NONZERO ($CODE) - killed, OOM or an unhandled error; see $LOG"
    echo "################ $JUDGE exited $CODE; continuing with the next judge"
  else
    note "SHELL    $JUDGE finished cleanly"
  fi
done

note "SHELL    overnight.sh done (worst exit code $STATUS)"
echo "################ done; summary at results/experiment04-judge_baseline/summary.csv"
echo "################ progress log at $RUN_LOG"
exit "$STATUS"
