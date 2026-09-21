#!/usr/bin/env bash
# Experiment 02: voting composition over independent voters. Detached, resumable, with a done marker.
# Usage: bash runs/experiment01-zeroshot_estimators/run.sh [--local-only] [extra run.py flags]
set -euo pipefail
cd "$(dirname "$0")/../.."

NAME=experiment01-zeroshot_estimators
mkdir -p "logs/$NAME"
rm -f "logs/$NAME/DONE"

nohup bash -c "python runs/$NAME/run.py $* && echo ok > logs/$NAME/DONE || echo failed > logs/$NAME/DONE" \
  > "logs/$NAME/run.log" 2>&1 &
echo "$!" > "logs/$NAME/run.pid"
echo "started $NAME (pid $!) - safe to disconnect"
echo "progress:  tail -f logs/$NAME/run.log"
echo "finished?  cat logs/$NAME/DONE"
echo "results:   cat results/$NAME/reports.json"
