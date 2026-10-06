#!/usr/bin/env bash
# GPU smoke test for the judge baseline: both views on a small slice, then a report of what landed on disk.
# Usage: bash runs/experiment04-judge_baseline/smoke.sh [task]        LIMIT=50 changes the slice size.
# JUDGE, PANEL, METHOD and MODES override the defaults; MATCH picks the cell family.
set -euo pipefail
cd "$(dirname "$0")/../.."

TASK=${1:-csqa}
JUDGE=${JUDGE:-vllm/g3-12i}
PANEL=${PANEL:-"vllm/g2-9i vllm/l31-8bi vllm/g3-12i"}
METHOD=${METHOD:-consistency_t0.7}
MODES=${MODES:-"verbalized"}
MATCH=${MATCH:-_cs7s}
LIMIT=${LIMIT:-20}
OUT=${OUT:-results/experiment04-judge_baseline}

for VIEW in reasoning reasoning_confidence; do
  echo "==================== $TASK  view=$VIEW ===================="
  python runs/experiment04-judge_baseline/run.py \
    --task "$TASK" --judge "$JUDGE" --panel $PANEL \
    --view "$VIEW" --confidence-method "$METHOD" --modes $MODES \
    --match "$MATCH" --limit "$LIMIT" --out-dir "$OUT"
done

echo "==================== saved under $OUT/$TASK ===================="
python - "$OUT/$TASK" <<'PY'
"""Show one saved verdict in full, then confirm every row carries a justification and the confidences."""
import json, sys
from pathlib import Path

for cell in sorted(Path(sys.argv[1]).iterdir()):
    rows = [json.loads(line) for line in (cell / "verdicts.jsonl").read_text().splitlines() if line.strip()]
    print(f"\n{cell.name}  ({len(rows)} rows, {sorted(p.name for p in cell.iterdir())})")
    first = rows[0]
    print(f"  final_answer  {first['final_answer']}   correct={first['correct']}")
    print(f"  justification {first['justification']!r}")
    for key in ("verbalized", "ptrue", "seqprob"):
        present = sum(r[key] is not None for r in rows)
        values = [r[key] for r in rows if r[key] is not None]
        mean = f"mean={sum(values) / len(values):.3f}" if values else "mean=-"
        print(f"  {key:<14}{present}/{len(rows)} present  {mean}")
    blank = [r["id"] for r in rows if not r["justification"]]
    print(f"  justifications {len(rows) - len(blank)}/{len(rows)} written" + (f"  blank: {blank[:5]}" if blank else ""))
    print(f"  response of the first row:\n    " + first["response"].replace("\n", "\n    ")[:600])
PY
