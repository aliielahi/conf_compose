# Uncertainty in LLM reasoning

Training-free signals that tell where and when a reasoning trace goes wrong, read from token probabilities and
rollouts of open models.

## Ideas

| Folder | Question | Signal | Evaluation |
|---|---|---|---|
| `cross_scoring` | Can a panel of other models find the first wrong step of a solution? | Teacher-forced surprise of each token under every scorer (worst token or window per step); the drop, across each step, of each scorer's belief in the panel's own answer | ProcessBench first-error F1, step AUROC (also length-matched), solution AUROC, vs a trained PRM |
| `belief_tracking` | Does a model's belief at each prefix track its true value, the share of rollouts from that prefix that end correct? | Forced-answer belief $b_t$ vs rollout value $V_t$ (a martingale); excess movement $\sum_t (b_{t+1}-b_t)^2 - [b_0(1-b_0) - b_T(1-b_T)]$ | Calibration of $b_t$ against $V_t$; AUROC of excess movement for wrong traces |
| `split_context` | When a long context is split between agents, does each know what it is missing, and do their confidences compose into the full-context one? | Per-agent confidence $p_A, p_B$ vs full-context $p_{AB}$; composition residual $\mathrm{logit}\,p_{AB} - [\mathrm{logit}\,p_A + \mathrm{logit}\,p_B - \mathrm{logit}\,p_\varnothing]$ | Calibration with partial context; accuracy and calibration of full context vs split + aggregation vs split + debate, on redundant, unique and synergistic splits (multi-hop facts in separate parts) |
| `plan_execution` | Is a model unsure about the plan or about carrying it out? | $H(A) = I(A;S) + H(A \mid S)$ over sampled traces, $S$ the strategy | Whether the execution share predicts the gain from majority voting |

Every confidence is also reported within questions, among traces of the same question.

## Layout

- `src/data/` dataset loaders, shared by all ideas
- `src/<idea>/` code of one idea
- `runs/<idea>/` outputs of that idea

## Setup

```bash
bash docker.sh up      # vLLM container with src/ on the Python path
bash docker.sh shell
```
