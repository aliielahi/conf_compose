#!/usr/bin/env bash
# Fill the inference store for one or more datasets: 7 models x 5 sampled voters, whole test split, no validation.
# Several tasks in one call load each model once instead of once per task.
# Usage: bash runs/inference/sweep_full.sh gpqa truthfulqa       DRY_RUN=1 lists what is missing and exits.
set -euo pipefail
cd "$(dirname "$0")/../.."

if [ "$#" -eq 0 ]; then
  echo "usage: bash runs/inference/sweep_full.sh <task> [task ...]" >&2
  exit 1
fi
TASKS="$*"
MODELS=${MODELS:-"vllm/q3-4bi vllm/q3-8bi vllm/l32-3bi vllm/l31-8bi vllm/g2-9i vllm/g3-12i vllm/phi4mii"}
VOTERS=${VOTERS:-5}
NAME=inference-$(echo "$TASKS" | tr ' ' '+')
FLAGS="--answer-temperature 0.7 --voters $VOTERS --n-val none --n-test all --all-signals"
mkdir -p "logs/$NAME"

if [ -n "${DRY_RUN:-}" ]; then
  python runs/inference/run_inference.py --tasks $TASKS --models $MODELS $FLAGS --dry-run
  exit 0
fi

rm -f "logs/$NAME/DONE"
nohup bash -c "python runs/inference/run_inference.py --tasks $TASKS --models $MODELS $FLAGS \
  && echo ok > logs/$NAME/DONE || echo failed > logs/$NAME/DONE" > "logs/$NAME/run.log" 2>&1 &
echo "$!" > "logs/$NAME/run.pid"
echo "started $NAME (pid $!) - safe to disconnect"
echo "progress:  tail -f logs/$NAME/run.log   |   cat current_run.log"
echo "finished?  cat logs/$NAME/DONE"
for task in $TASKS; do echo "cells:     ls results/inferences/$task/"; done
