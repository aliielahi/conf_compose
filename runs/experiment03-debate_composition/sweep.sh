#!/usr/bin/env bash
# Debate inferences: the 15 voting groups x datasets, round 0 from the store, revision rounds, candidate scoring.
# Detaches itself, so closing the laptop or dropping ssh cannot kill it; rerunning resumes where it stopped.
# Everything for this experiment lands in logs/experiment03-debate_composition/:
#   current_run.log   progress, ETA, crashes, blocked cells and a summary     run.log   detached stdout
#   <stage>_round<r>_<model>.log   one per model step                         run.pid   for stop    DONE   on finish
# Usage: bash runs/experiment03-debate_composition/sweep.sh          start, detached
#        bash runs/experiment03-debate_composition/sweep.sh stop     stop a running sweep
#   TASKS="gpqa csqa"   GROUPS="1 2 3"   ROUNDS=2   LIMIT=20   STAGE=generate|score
#   EXTRA="--dry-run"   print the plan and exit       EXTRA="--dry-run --count-tokens"   also count prompt tokens
#   FOREGROUND=1        do not detach
set -uo pipefail
cd "$(dirname "$0")/../.."

NAME=experiment03-debate_composition
LOG_DIR=logs/$NAME
RUN_LOG=$LOG_DIR/current_run.log
PIDFILE=$LOG_DIR/run.pid
TASKS=${TASKS:-"gpqa truthfulqa csqa gsm8k boolq"}
ROUNDS=${ROUNDS:-1}
STAGE=${STAGE:-all}
mkdir -p "$LOG_DIR"

case "${EXTRA:-}" in *--dry-run*) DRY=1 ;; esac
note() { [ -n "${DRY:-}" ] || echo "$(date '+%Y-%m-%d %H:%M:%S')  $*" >> "$RUN_LOG"; }

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

[ -n "${DRY:-}" ] && FOREGROUND=1

if [ -z "${SWEEP_DETACHED:-}" ] && [ "${FOREGROUND:-0}" != "1" ]; then
  if [ -s "$PIDFILE" ] && kill -0 "$(cut -d' ' -f1 "$PIDFILE")" 2> /dev/null; then
    echo "already running as pid $(cut -d' ' -f1 "$PIDFILE"); stop it first" >&2; exit 1
  fi
  rm -f "$PIDFILE" "$LOG_DIR/DONE"
  export SWEEP_DETACHED=1 TASKS ROUNDS STAGE GROUPS="${GROUPS:-}" LIMIT="${LIMIT:-}" EXTRA="${EXTRA:-}"
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
  echo "  stop     : bash runs/experiment03-debate_composition/sweep.sh stop"
  exit 0
fi

ARGS="--tasks $TASKS --rounds $ROUNDS --stage $STAGE ${GROUPS:+--groups $GROUPS} ${LIMIT:+--limit $LIMIT} ${EXTRA:-}"
if [ -n "${DRY:-}" ]; then
  exec python runs/experiment03-debate_composition/run.py $ARGS
fi

echo "$$ $(ps -o pgid= -p $$ | tr -d ' ')" > "$PIDFILE"
trap 'note "SHELL    received a signal, exiting"; echo stopped > "$LOG_DIR/DONE"; rm -f "$PIDFILE"; exit 143' INT TERM

note "SHELL    sweep.sh starting (pid $$): tasks=$TASKS rounds=$ROUNDS stage=$STAGE groups=${GROUPS:-all} limit=${LIMIT:-full}"
python runs/experiment03-debate_composition/run.py $ARGS
STATUS=$?
note "SHELL    sweep.sh done (exit $STATUS)"
[ "$STATUS" -eq 0 ] && echo ok > "$LOG_DIR/DONE" || echo "failed $STATUS" > "$LOG_DIR/DONE"
rm -f "$PIDFILE"
exit "$STATUS"
