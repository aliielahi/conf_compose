# experiment03 — debate inferences

**What.** The 15 voting groups debate synchronously. Round 0 is read from `results/inferences/` (the exact
`s70v0_*_cs7s` cells voting used, hashed in each cell's `settings.json`); only rounds >= 1 are generated.
In round r every agent sees the question, its own round r-1 response, and every peer's round r-1 answer and
reasoning, under neutral names in a shuffled order, and writes an updated answer and explanation. Agents do
not have to agree; aggregating them is a later step.

**Confidence.** The same estimators as voting, run by round 0's own pipeline on the exact context each agent
saw: consistency (5 resamples at 0.7) and the sequence-probability family. After the last round every agent
scores every candidate (labels, every answer any agent proposed in any round, and their resamples) in the
`direct` and `reasoned` contexts plus content-free baselines, so `cand_direct_sum` and its variants are
computed downstream exactly as in voting. Verbalized and verification signals are skipped.

**Answers.** The revision prompt asks for the final answer in full. A revision that names a peer it agrees
with, or keeps its own answer without restating it, is still resolved, and `answer_source` records how.

```bash
bash runs/experiment03-debate_composition/sweep.sh                     # all tasks, 15 groups, 1 round
EXTRA="--dry-run --count-tokens" bash runs/experiment03-debate_composition/sweep.sh   # plan + prompt lengths
TASKS=csqa GROUP_IDS="1 6 14" ROUNDS=2 LIMIT=20 bash runs/experiment03-debate_composition/sweep.sh   # smoke
python runs/experiment03-debate_composition/check.py --match _n20          # inspect
```

Outputs: `results/debate_inferences/<task>/<group>--s70v0[_n<limit>]--<digest>/` with `settings.json`,
`round_<r>/<model>.jsonl` (inference-store record schema plus the messages, peer order and answer
resolution) and `candidates_through_round_<R>/<model>.jsonl`. Logs: `logs/experiment03-debate_composition/`.
