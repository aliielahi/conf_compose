#!/usr/bin/env bash
# Every model subset of the requested sizes: one CSV row per (subset, dataset), each voting on its own majority.
# Usage: bash runs/experiment02-voting_composition/sweep_atomic.sh      DRY_RUN=1 lists the groups and exits.
set -euo pipefail
cd "$(dirname "$0")/../.."

export MODELS=${MODELS:-"vllm/q3-4bi vllm/q3-8bi vllm/l32-3bi vllm/l31-8bi vllm/g2-9i vllm/g3-12i vllm/phi4mii"}
export TASKS=${TASKS:-"csqa gsm8k truthfulqa gpqa boolq"}
export SIZES=${SIZES:-"2 3 4 5 6"}
export CSV=${CSV:-results/voting_atomic/atomic.csv}
NAME=voting_atomic
mkdir -p "logs/$NAME" "$(dirname "$CSV")"

subsets() {
  python -c '
import os
from itertools import combinations
models = os.environ["MODELS"].split()
for size in (int(s) for s in os.environ["SIZES"].split()):
    for combo in combinations(models, size):
        print(" ".join(combo))
'
}

if [ "${1:-}" = "--child" ]; then
  total=$(subsets | grep -c .)
  index=0
  while read -r group; do
    index=$((index + 1))
    echo "[$(date +%H:%M:%S)] $index/$total  $group"
    if ! python runs/experiment02-voting_composition/atomic.py --tasks $TASKS --group $group \
        --out "$CSV" > /dev/null; then
      echo "FAILED on: $group"
      echo failed > "logs/$NAME/DONE"
      exit 1
    fi
  done < <(subsets)
  echo "[$(date +%H:%M:%S)] all $total group(s) done"
  echo ok > "logs/$NAME/DONE"
  exit 0
fi

echo "$(subsets | grep -c .) group(s) over $(echo $TASKS | wc -w | tr -d ' ') task(s) -> $CSV"
if [ -n "${DRY_RUN:-}" ]; then
  subsets | nl
  exit 0
fi

rm -f "logs/$NAME/DONE"
nohup bash "$0" --child > "logs/$NAME/run.log" 2>&1 &
echo "$!" > "logs/$NAME/run.pid"
echo "started (pid $!) - safe to disconnect"
echo "progress:  tail -f logs/$NAME/run.log"
echo "finished?  cat logs/$NAME/DONE"
echo "results:   $CSV"
