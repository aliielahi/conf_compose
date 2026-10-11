#!/usr/bin/env bash
# Debate composition with sequence-probability confidence, one background run per variant, both rounds.
# Usage: bash paper_results/seq_vs_cons/run.sh    then, when every DONE file says ok: python paper_results/seq_vs_cons/extract.py
set -euo pipefail
cd "$(dirname "$0")/../.."

STORE=${STORE:-results/debate_inferences_fresh/debate_inferences}
LOGS=paper_results/seq_vs_cons/logs
# name context score: the same variants as VARIANTS in extract.py.
VARIANTS="seq_label:direct:norm_sum seq_token:direct:norm_mean seq_reasoned:reasoned:norm_sum"

for variant in $VARIANTS; do
  IFS=: read -r name context score <<< "$variant"
  mkdir -p "$LOGS/$name"
  rm -f "$LOGS/$name/DONE"
  nohup bash -c "${PYTHON:-python} runs/experiment05-debate_comp/run.py --rounds 1 2 --estimators seq \
      --context $context --seq-score $score --debate-store $STORE --require-complete \
      && echo ok > $LOGS/$name/DONE || echo failed > $LOGS/$name/DONE" > "$LOGS/$name/run.log" 2>&1 &
  echo "$name started (pid $!), log: $LOGS/$name/run.log"
done
echo "finished?  cat $LOGS/*/DONE"
