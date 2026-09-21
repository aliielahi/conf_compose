#!/usr/bin/env bash
# Inner loop of sweep05; run it through runs/sweeps/sweep05-panels.sh, not directly.
set -uo pipefail
cd "$(dirname "$0")/../.."

for task in $TASKS; do
  for target in $TARGETS; do
    echo "[$(date +%H:%M:%S)] start $task/$target"
    python runs/composition/run_panels.py --task "$task" --target "$target" \
      --validation "$SOURCE/$task/*/validation.jsonl" --test "$SOURCE/$task/*/test.jsonl" \
      --target-scores "$SOURCE/$task/target_verification_*.json" \
      --families $FAMILIES --n-boot "$NBOOT" --cap "$CAP" \
      > "logs/$SWEEP/${task}_${target}.log" 2>&1
    if [ $? -ne 0 ]; then
      echo "[$(date +%H:%M:%S)] FAILED $task/$target - see logs/$SWEEP/${task}_${target}.log"
      echo "failed" > "logs/$SWEEP/DONE"
      exit 1
    fi
    echo "[$(date +%H:%M:%S)] ok $task/$target"
  done
done

python runs/composition/summarize_panels.py --metric auroc --rule mean > "logs/$SWEEP/summary.txt" 2>&1
echo "[$(date +%H:%M:%S)] all cells done"
echo "ok" > "logs/$SWEEP/DONE"
