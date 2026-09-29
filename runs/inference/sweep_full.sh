#!/usr/bin/env bash
# Fill the inference store for one or more datasets: 7 models x 5 sampled voters, whole test split, no validation.
# Several tasks in one call load each model once instead of once per task.
# Usage: bash runs/inference/sweep_full.sh gpqa truthfulqa       DRY_RUN=1 lists what is missing and exits.
# RETRY=2048 regenerates only the rows that hit the old ceiling, reusing everything else from cache.
# CANDIDATES=1 scores every candidate answer under every model; SAMPLE_LOGPROBS=1 keeps per-resample scores.
set -euo pipefail
cd "$(dirname "$0")/../.."

if [ "$#" -eq 0 ]; then
  echo "usage: bash runs/inference/sweep_full.sh <task> [task ...]" >&2
  exit 1
fi
TASKS="$*"
MODELS=${MODELS:-"vllm/q3-4bi vllm/q3-8bi vllm/l32-3bi vllm/l31-8bi vllm/g2-9i vllm/g3-12i vllm/phi4mii"}
VOTERS=${VOTERS:-5}
NAME=inference-$(echo "$TASKS" | tr ' ' '+')${RETRY:+-r$RETRY}${CANDIDATES:+-cs}${SAMPLE_LOGPROBS:+-lp}
FLAGS="--answer-temperature 0.7 --voters $VOTERS --n-val none --n-test all --all-signals"
[ -n "${RETRY:-}" ] && FLAGS="$FLAGS --retry-max-tokens $RETRY"
[ -n "${CANDIDATES:-}" ] && FLAGS="$FLAGS --score-candidates --candidate-samples"
[ -n "${SAMPLE_LOGPROBS:-}" ] && FLAGS="$FLAGS --consistency-logprobs"
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
