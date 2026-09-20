#!/usr/bin/env bash
# Independent voters: each model answers VOTERS times at T=0.7, each answer with its own consistency samples.
# Phase 2 (separate script) then verifies the anchor and majority-vote answers with every model's verifier.
# Usage: VOTERS=5 bash runs/sweeps/sweep04-voters.sh        DRY_RUN=1 to list pending work and exit.
set -euo pipefail
cd "$(dirname "$0")/../.."

SWEEP=${SWEEP:-sweep04}
VOTERS=${VOTERS:-5}
MODELS=${MODELS:-"vllm/q3-4bi vllm/l31-8bi vllm/g3-12i vllm/phi4mii"}
TASKS=${TASKS:-"csqa boolq"}
DRY_RUN=${DRY_RUN:-}
mkdir -p "logs/$SWEEP" "results/$SWEEP"

models=$(echo $MODELS | wc -w | tr -d ' ')
tasks=$(echo $TASKS | wc -w | tr -d ' ')
echo "sweep $SWEEP: $VOTERS voters x $models models x $tasks tasks, answers sampled at T=0.7"
echo "about $((models * tasks * VOTERS * 700 * 6)) full generations (700 examples, 1 answer + 5 samples each)"
echo "lower VOTERS to cut it; the same-model panel then caps at that many voters"

if [ -n "$DRY_RUN" ]; then
  python runs/zeroshot_baseline/run_baselines.py --sweep "$SWEEP" --voters "$VOTERS" \
    --models $MODELS --tasks $TASKS --dry-run
  exit 0
fi

nohup python runs/zeroshot_baseline/run_baselines.py --sweep "$SWEEP" --voters "$VOTERS" \
  --models $MODELS --tasks $TASKS \
  -- --answer-temperature 0.7 --no-verbalized --execution "$SWEEP" "$@" \
  > "logs/$SWEEP/sweep.log" 2>&1 &
echo "$!" > "logs/$SWEEP/sweep.pid"
echo
echo "started (pid $!) | progress: tail -f logs/$SWEEP/sweep.log"
echo "then phase 2: bash runs/sweeps/sweep04-score_targets.sh"
echo "then offline:  python runs/composition/run_panels.py --task csqa --target majority \\"
echo "                 --validation 'results/$SWEEP/csqa/*/validation.jsonl' --test 'results/$SWEEP/csqa/*/test.jsonl'"
