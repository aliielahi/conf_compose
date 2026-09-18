#!/usr/bin/env bash
# Sweep 02: CommonsenseQA + BoolQ on the baseline models, and ProntoQA on weaker models, detached with nohup.
# Usage: bash runs/sweeps/sweep02-newtasks.sh [-- extra eval_zero_shot.py flags]
set -euo pipefail
cd "$(dirname "$0")/../.."

SWEEP=sweep02
RUNNER=runs/zeroshot_baseline/run_baselines.py
mkdir -p "logs/$SWEEP" "results/$SWEEP"
nohup bash -c "python $RUNNER --sweep $SWEEP --tasks csqa boolq $* && \
               python $RUNNER --sweep $SWEEP --tasks prontoqa --models vllm/l32-3bi vllm/phi4mii $*" \
      > "logs/$SWEEP/sweep.log" 2>&1 &
echo "$!" > "logs/$SWEEP/sweep.pid"
echo "started $SWEEP (pid $!)"
echo "progress: tail -f logs/$SWEEP/sweep.log"
echo "summary:  python runs/zeroshot_baseline/summarize.py --sweep $SWEEP"
