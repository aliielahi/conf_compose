# Uncertainty in LLM reasoning

Where and when a reasoning trace goes wrong, what models know about it, and what happens when that uncertainty is
shared between agents or used as a reward.

## Ideas

| Folder | Question | Signal | Evaluation | Status |
|---|---|---|---|---|
| `cross_scoring` | Can a panel of other models find the first wrong step of a solution? Does a reader have to be able to solve the problem to notice the error (`readers.py`)? | Teacher-forced surprise of each token under every scorer (worst token, window or step mean); the drop, across each step, of each scorer's belief in the panel's own answer | ProcessBench first-error F1, step AUROC (also length-matched), solution AUROC, vs a trained PRM; localisation of the first error by readers that solved vs did not (solution and reader fixed effects), and by shared vs reader-specific signal | Dead as a detector. Belief drop is best (GSM8K length-matched step AUROC 0.72, F1 0.46; MATH 0.63, 0.35), far below trained PRMs. Readers: 84–86% of surprise is shared across readers and solving the problem doesn't change it (fixed effect ≈ 0 ± 0.01); belief drop +0.05 ± 0.02 on GSM8K, n.s. on MATH |
| `belief_tracking` | Does a model's belief at each prefix track its true value, the share of rollouts from that prefix that end correct? | Forced-answer belief $b_t$ vs rollout value $V_t$ (a martingale); excess movement $\sum_t (b_{t+1}-b_t)^2 - [b_0(1-b_0) - b_T(1-b_T)]$ | Calibration of $b_t$ against $V_t$; AUROC of excess movement for wrong traces | Dead as a detector. $b_t$ tracks progress, not value (0.10 vs 0.81 before thinking); self-consistency reaches 0.90–0.96 |
| `value_collapse` | Does a solution's chance of success collapse before, at or after its first labelled error? | Rollout value $V_k$ from every ProcessBench prefix; belief $b_k$ from `cross_scoring` | Lag between the largest drop in $V$ and the labelled error; share of labelled errors after which $V$ recovers | Next |
| `split_context` | With evidence split between agents, how do incentives (shared goal, mixed, opposed advocates with a judge) change what is disclosed and how calibrated the group is? | Group log loss split into disclosure loss (facts never shared), integration loss (shared, not used) and distortion (stated falsely) | Accuracy, Brier and the three losses per condition, on split multi-hop QA and HiddenBench | Literature review done |
| `label_free_reward` | How biased is the policy gradient when a label-free reward (majority vote, vote share, forced belief) replaces correctness? | Excess Brier of the reward against correctness or value, which bounds the gradient bias | Offline on `belief_tracking` traces (advantage sign agreement, predicted bias); then GRPO on a 1.5B model | Offline diagnostic written (`diagnose.py`), to run |
| `performative_confidence` | Trained with a Brier reward on confidence stated mid-reasoning, does a model learn to steer its reasoning so the confidence comes true? | Stated confidence vs rollout value $V_t$; how much the stated confidence changes the outcome | GRPO on a small model; confidence stated mid-reasoning vs after the answer | Not run |
| `group_verifier` | Can a small trained model read several models' traces and output a calibrated group confidence? | Calibrated P(correct) from all traces | Brier and AUROC vs majority vote and the best single model's confidence | Not run, no prior work found |
| `plan_execution` | Is a model unsure about the plan or about carrying it out? | $H(A) = I(A;S) + H(A \mid S)$ over sampled traces, $S$ the strategy | Whether the execution share predicts the gain from majority voting | Not run, traces exist |

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
