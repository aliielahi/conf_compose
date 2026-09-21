# experiment03 — debate composition

**Question.** Once agents have read each other, does interaction change how useful their confidence evidence
is, and does composition over debate rounds beat composition over independent voters?

**Datasets.** GSM8K for now, where the traces exist. **Models.** Two agents, one round.

Debate produces conversation traces rather than zero-shot records, so it does not go through
`results/inferences/`; generation and confidence are two passes over a trace file.

```bash
python runs/experiment03-debate_composition/run_debate.py --task gsm8k ...
python runs/experiment03-debate_composition/add_confidence.py ...
python runs/experiment03-debate_composition/inspect_debate.py ...
```

Outputs land in `results/experiment03-debate_composition/`.
