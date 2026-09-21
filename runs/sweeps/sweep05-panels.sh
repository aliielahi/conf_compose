#!/usr/bin/env bash
# Offline voting panels over the voter sweep: every task x target cell, detached, with a done marker.
# Usage: bash runs/sweeps/sweep05-panels.sh      (no GPU; safe to disconnect)
set -euo pipefail
cd "$(dirname "$0")/../.."

export SWEEP=${SWEEP:-sweep05}
export SOURCE=${SOURCE:-results/sweep04}
export TASKS=${TASKS:-"csqa boolq"}
export TARGETS=${TARGETS:-"anchor majority"}
export NBOOT=${NBOOT:-2000}
export CAP=${CAP:-5}
export FAMILIES=${FAMILIES:-"cons ver cons+ver vertgt cons+vertgt"}

mkdir -p "logs/$SWEEP"
rm -f "logs/$SWEEP/DONE"
nohup bash runs/sweeps/_sweep05_body.sh > "logs/$SWEEP/sweep.log" 2>&1 &
echo "$!" > "logs/$SWEEP/sweep.pid"
echo "started $SWEEP (pid $!) - safe to disconnect"
echo "progress:  tail -f logs/$SWEEP/sweep.log"
echo "finished?  cat logs/$SWEEP/DONE     (ok = all four cells done)"
echo "results:   cat logs/$SWEEP/summary.txt"
