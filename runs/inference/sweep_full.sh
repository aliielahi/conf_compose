#!/usr/bin/env bash
# Fill the inference store for ONE dataset: 7 models x 5 sampled voters, whole test split, no validation.
# Usage: bash runs/inference/sweep_full.sh gsm8k          DRY_RUN=1 to list what is missing and exit.
set -euo pipefail
cd "$(dirname "$0")/../.."

TASK=${1:?usage: bash runs/inference/sweep_full.sh <task>}
MODELS=${MODELS:-"vllm/q3-4bi vllm/q3-8bi vllm/l32-3bi vllm/l31-8bi vllm/g2-9i vllm/g3-12i vllm/phi4mii"}
VOTERS=${VOTERS:-5}
NAME=inference-$TASK
mkdir -p "logs/$NAME"

if [ -n "${DRY_RUN:-}" ]; then
  python runs/inference/run_inference.py --tasks "$TASK" --models $MODELS \
    --answer-temperature 0.7 --voters "$VOTERS" --n-val none --n-test all --all-signals --dry-run
  exit 0
fi

rm -f "logs/$NAME/DONE"
nohup bash -c "python runs/inference/run_inference.py --tasks '$TASK' --models $MODELS \
    --answer-temperature 0.7 --voters $VOTERS --n-val none --n-test all --all-signals \
  && echo ok > logs/$NAME/DONE || echo failed > logs/$NAME/DONE" > "logs/$NAME/run.log" 2>&1 &
echo "$!" > "logs/$NAME/run.pid"
echo "started $NAME (pid $!) - safe to disconnect"
echo "progress:  tail -f logs/$NAME/run.log"
echo "finished?  cat logs/$NAME/DONE"
echo "cells:     ls results/inferences/$TASK/"
