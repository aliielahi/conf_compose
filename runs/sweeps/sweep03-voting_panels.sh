#!/usr/bin/env bash
# Voting panels: panel size x model set (homo/hetero) x estimator family, on both closed-label tasks.
# Usage: bash runs/sweeps/sweep03-voting_panels.sh [--n-boot 2000] [-- extra run_panels.py flags]
set -euo pipefail
cd "$(dirname "$0")/../.."

SWEEP=sweep03
SOURCE=${SOURCE:-results/sweep02}
mkdir -p "logs/$SWEEP"
for task in csqa boolq; do
  for target in majority anchor; do
    echo "=== $task / $target ==="
    python runs/composition/run_panels.py --task "$task" --target "$target" \
      --validation "$SOURCE/$task/*/validation.jsonl" --test "$SOURCE/$task/*/test.jsonl" \
      "$@" 2>&1 | tee "logs/$SWEEP/${task}_${target}.log"
  done
done
echo "summary: python runs/composition/summarize_panels.py"
