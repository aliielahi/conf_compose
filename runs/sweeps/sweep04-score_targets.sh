#!/usr/bin/env bash
# Phase 2 of sweep04: every model verifies the anchor answer and the majority-vote answer of the voter panel.
# Usage: bash runs/sweeps/sweep04-score_targets.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

SWEEP=${SWEEP:-sweep04}
MODELS=${MODELS:-"vllm/q3-4bi vllm/l31-8bi vllm/g3-12i vllm/phi4mii"}
TASKS=${TASKS:-"csqa boolq"}
mkdir -p "logs/$SWEEP"

for task in $TASKS; do
  for model in $MODELS; do
    tag=$(echo "$model" | tr '/' '_')
    echo "=== $task / $model ==="
    python runs/composition/score_targets.py --task "$task" --model "$model" \
      --root "results/$SWEEP/$task" 2>&1 | tee "logs/$SWEEP/targets_${task}_${tag}.log"
  done
done
echo "done: results/$SWEEP/<task>/target_verification_<model>_<split>.json"
