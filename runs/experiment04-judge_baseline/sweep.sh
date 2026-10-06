#!/usr/bin/env bash
# Judge sweep: 15 panels x datasets x 2 judges x (reasoning control + 2 confidence estimators).
# Detaches itself, so closing the laptop or dropping ssh cannot kill it.
# One judge per process, because two vLLM engines cannot share one GPU at 0.7 utilisation each.
# Datasets run cheapest first; rerunning redoes only cells that are missing or from an older context window.
# Everything for this experiment lands in logs/experiment04-judge_baseline/:
#   current_run.log   progress, ETA, crashes and a summary per launch      run.log   detached stdout
#   <judge>.log       full stdout of one judge's process                   run.pid   for stop    DONE   on finish
# Usage: bash runs/experiment04-judge_baseline/sweep.sh          start, detached
#        bash runs/experiment04-judge_baseline/sweep.sh stop     stop a running sweep
#   TASKS="gpqa csqa"   JUDGES="vllm/g3-27i"   SIZES="2 3"   FORCE=1 redo complete cells
#   EXTRA="--dry-run" print the plan       EXTRA="--no-control" drop the control arm      FOREGROUND=1
set -uo pipefail
cd "$(dirname "$0")/../.."

NAME=experiment04-judge_baseline
LOG_DIR=logs/$NAME
RUN_LOG=$LOG_DIR/current_run.log
PIDFILE=$LOG_DIR/run.pid
JUDGES=${JUDGES:-"vllm/g3-27i vllm/l32-3bi"}
TASKS=${TASKS:-"gpqa truthfulqa csqa gsm8k boolq"}
mkdir -p "$LOG_DIR"

note() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $*" >> "$RUN_LOG"; }

# Children first, so bash is not left waiting on a python that outlives the signal.
tree_kill() {
  local parent=$1 signal=$2 child
  for child in $(pgrep -P "$parent" 2> /dev/null); do tree_kill "$child" "$signal"; done
  kill "-$signal" "$parent" 2> /dev/null || true
}

if [ "${1:-}" = "stop" ]; then
  [ -s "$PIDFILE" ] || { echo "no $PIDFILE; nothing to stop" >&2; exit 1; }
  read -r PID PGID < "$PIDFILE"
  if ! kill -0 "$PID" 2> /dev/null; then
    echo "pid $PID is already gone"; rm -f "$PIDFILE"; exit 0
  fi
  # Only a pid that leads its own group is safe to signal as a group.
  if [ "$PID" = "${PGID:-}" ]; then kill -TERM -- "-$PGID" 2> /dev/null || true; else tree_kill "$PID" TERM; fi
  for _ in 1 2 3 4 5 6 7 8 9 10; do kill -0 "$PID" 2> /dev/null || break; sleep 1; done
  if kill -0 "$PID" 2> /dev/null; then
    if [ "$PID" = "${PGID:-}" ]; then kill -KILL -- "-$PGID" 2> /dev/null || true; else tree_kill "$PID" KILL; fi
  fi
  note "SHELL    stopped by request (pid $PID)"
  rm -f "$PIDFILE"; echo "stopped $PID"; exit 0
fi

case "${EXTRA:-}" in *--dry-run*) FOREGROUND=1 ;; esac

if [ -z "${SWEEP_DETACHED:-}" ] && [ "${FOREGROUND:-0}" != "1" ]; then
  if [ -s "$PIDFILE" ] && kill -0 "$(cut -d' ' -f1 "$PIDFILE")" 2> /dev/null; then
    echo "already running as pid $(cut -d' ' -f1 "$PIDFILE"); stop it first" >&2; exit 1
  fi
  rm -f "$PIDFILE" "$LOG_DIR/DONE"
  export SWEEP_DETACHED=1 JUDGES TASKS SIZES="${SIZES:-}" EXTRA="${EXTRA:-}" FORCE="${FORCE:-}"
  if command -v setsid > /dev/null 2>&1; then
    setsid bash "$0" < /dev/null > "$LOG_DIR/run.log" 2>&1 &
  else
    nohup bash "$0" < /dev/null > "$LOG_DIR/run.log" 2>&1 &
  fi
  disown 2> /dev/null || true
  for _ in 1 2 3 4 5 6 7 8 9 10; do [ -s "$PIDFILE" ] && break; sleep 1; done
  echo "started $NAME as pid $(cut -d' ' -f1 "$PIDFILE" 2> /dev/null || echo '?') - safe to disconnect"
  echo "  progress : tail -f $RUN_LOG"
  echo "  stdout   : tail -f $LOG_DIR/run.log"
  echo "  finished : cat $LOG_DIR/DONE"
  echo "  stop     : bash runs/experiment04-judge_baseline/sweep.sh stop"
  exit 0
fi

echo "$$ $(ps -o pgid= -p $$ | tr -d ' ')" > "$PIDFILE"
trap 'note "SHELL    received a signal, exiting"; echo stopped > "$LOG_DIR/DONE"; rm -f "$PIDFILE"; exit 143' INT TERM

note "$(printf '=%.0s' {1..78})"
note "SHELL    sweep.sh starting (pid $$): judges=$JUDGES tasks=$TASKS"

STATUS=0
for JUDGE in $JUDGES; do
  LOG="$LOG_DIR/$(echo "$JUDGE" | tr '/' '-').log"
  echo "################ $JUDGE  ->  $LOG"
  note "SHELL    launching $JUDGE, stdout at $LOG"
  python runs/experiment04-judge_baseline/sweep.py \
    --judges "$JUDGE" --tasks $TASKS --log "$RUN_LOG" \
    ${SIZES:+--panel-sizes $SIZES} ${FORCE:+--force} ${EXTRA:-} 2>&1 | tee -a "$LOG"
  CODE=${PIPESTATUS[0]}
  if [ "$CODE" -ne 0 ]; then
    STATUS=$CODE
    note "SHELL    $JUDGE EXITED NONZERO ($CODE) - killed, OOM or an unhandled error; see $LOG"
  else
    note "SHELL    $JUDGE finished cleanly"
  fi
done

note "SHELL    sweep.sh done (worst exit code $STATUS)"
[ "$STATUS" -eq 0 ] && echo ok > "$LOG_DIR/DONE" || echo "failed $STATUS" > "$LOG_DIR/DONE"
rm -f "$PIDFILE"
exit "$STATUS"
