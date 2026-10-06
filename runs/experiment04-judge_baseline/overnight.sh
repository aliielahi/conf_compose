#!/usr/bin/env bash
# Overnight judge sweep: 15 panels x 5 datasets x 2 judges x (reasoning control + 2 confidence estimators).
# Detaches itself by default, so closing the laptop or dropping the ssh session cannot kill it.
# One judge per process, because two vLLM engines cannot share one GPU at 0.7 utilisation each.
# Datasets run cheapest first, so an interrupted night leaves whole datasets finished.
# Rerunning skips every cell already on disk, so this is the resume command too.
# Progress, crashes and a summary land in logs/current_run.log; each judge also gets its own stdout log.
# Usage: bash runs/experiment04-judge_baseline/overnight.sh          detaches and returns the pid
#        bash runs/experiment04-judge_baseline/overnight.sh stop     stop a detached run
#   TASKS="gpqa csqa"      only these datasets          JUDGES="vllm/g3-27i"   only this judge
#   SIZES="2 3"            only panels of these sizes   EXTRA="--no-control"   drop the control arm
#   EXTRA="--dry-run"      print the plan and exit      FOREGROUND=1           do not detach
set -uo pipefail
cd "$(dirname "$0")/../.."

JUDGES=${JUDGES:-"vllm/g3-27i vllm/l32-3bi"}
TASKS=${TASKS:-"gpqa truthfulqa csqa gsm8k boolq"}
LOGS=${LOGS:-logs}
RUN_LOG=$LOGS/current_run.log
PIDFILE=$LOGS/overnight.pid
mkdir -p "$LOGS"

note() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $*" >> "$RUN_LOG"; }

# Children first, so bash is not left waiting on a python that outlives the signal.
tree_kill() {
  local parent=$1 signal=$2 child
  for child in $(pgrep -P "$parent" 2> /dev/null); do tree_kill "$child" "$signal"; done
  kill "-$signal" "$parent" 2> /dev/null || true
}

# The pidfile holds "pid pgid"; only a pid that leads its own group is safe to signal as a group.
if [ "${1:-}" = "stop" ]; then
  [ -s "$PIDFILE" ] || { echo "no $PIDFILE; nothing to stop" >&2; exit 1; }
  read -r PID PGID < "$PIDFILE"
  if ! kill -0 "$PID" 2> /dev/null; then
    echo "pid $PID is already gone"
    rm -f "$PIDFILE"
    exit 0
  fi
  if [ "$PID" = "${PGID:-}" ]; then
    kill -TERM -- "-$PGID" 2> /dev/null || true
  else
    tree_kill "$PID" TERM
  fi
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    kill -0 "$PID" 2> /dev/null || break
    sleep 1
  done
  if kill -0 "$PID" 2> /dev/null; then
    [ "$PID" = "${PGID:-}" ] && kill -KILL -- "-$PGID" 2> /dev/null || tree_kill "$PID" KILL
  fi
  note "SHELL    stopped by request (pid $PID)"
  rm -f "$PIDFILE"
  echo "stopped $PID"
  exit 0
fi

# Re-exec detached in a new session, so no hangup from a closed terminal reaches it.
# A dry run prints a plan, so there is nothing to detach from.
case "${EXTRA:-}" in *--dry-run*) FOREGROUND=1 ;; esac

if [ -z "${SWEEP_DETACHED:-}" ] && [ "${FOREGROUND:-0}" != "1" ]; then
  if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2> /dev/null; then
    echo "already running as pid $(cat "$PIDFILE"); stop it first or pass FOREGROUND=1" >&2
    exit 1
  fi
  BOOT=$LOGS/overnight-$(date +%Y%m%d_%H%M).log
  rm -f "$PIDFILE"
  export SWEEP_DETACHED=1 JUDGES TASKS LOGS SIZES="${SIZES:-}" EXTRA="${EXTRA:-}"
  if command -v setsid > /dev/null 2>&1; then
    setsid bash "$0" < /dev/null > "$BOOT" 2>&1 &
  else
    nohup bash "$0" < /dev/null > "$BOOT" 2>&1 &
  fi
  disown 2> /dev/null || true
  # The child writes its own process group, which is what stop needs; wait for it to appear.
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    [ -s "$PIDFILE" ] && break
    sleep 1
  done
  echo "detached as pid $(cut -d' ' -f1 "$PIDFILE" 2> /dev/null || echo '?')"
  echo "  progress : tail -f $RUN_LOG"
  echo "  stdout   : tail -f $BOOT"
  echo "  stop     : bash runs/experiment04-judge_baseline/overnight.sh stop"
  exit 0
fi

echo "$$ $(ps -o pgid= -p $$ | tr -d ' ')" > "$PIDFILE"
trap 'note "SHELL    received a signal, exiting"; rm -f "$PIDFILE"; exit 143' INT TERM

note "$(printf '=%.0s' {1..78})"
note "SHELL    overnight.sh starting (pid $$): judges=$JUDGES tasks=$TASKS"

STATUS=0
for JUDGE in $JUDGES; do
  TAG=$(echo "$JUDGE" | tr '/' '-')
  LOG="$LOGS/judge_sweep-$TAG-$(date +%Y%m%d_%H%M).log"
  echo "################ $JUDGE  ->  $LOG"
  note "SHELL    launching $JUDGE, stdout at $LOG"
  python runs/experiment04-judge_baseline/sweep.py \
    --judges "$JUDGE" --tasks $TASKS --log "$RUN_LOG" \
    ${SIZES:+--panel-sizes $SIZES} ${EXTRA:-} 2>&1 | tee "$LOG"
  CODE=${PIPESTATUS[0]}
  if [ "$CODE" -ne 0 ]; then
    STATUS=$CODE
    note "SHELL    $JUDGE EXITED NONZERO ($CODE) - killed, OOM or an unhandled error; see $LOG"
    echo "################ $JUDGE exited $CODE; continuing with the next judge"
  else
    note "SHELL    $JUDGE finished cleanly"
  fi
done

note "SHELL    overnight.sh done (worst exit code $STATUS)"
rm -f "$PIDFILE"
echo "################ done; summary at results/experiment04-judge_baseline/summary.csv"
echo "################ progress log at $RUN_LOG"
exit "$STATUS"
