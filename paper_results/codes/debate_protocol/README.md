# Post-debate consistency report

Run the composition sweep after all round-one consistency samples are saved:

```bash
python runs/experiment05-debate_comp/run.py --estimators cons --require-complete
python runs/experiment05-debate_comp/check.py --run results/debate_composition/run_<id>
python paper_results/codes/debate_protocol/run.py --run results/debate_composition/run_<id>
```

The report writes to `paper_results/results/debate_protocol/run_<id>/cons/`. `atomic.csv` has one row per dataset, model group, and method. Each table has a plain-text and LaTeX version. `all_*` averages over the model groups within each dataset; `size_2_*` through `size_6_*` restrict to a group size. Absolute tables show metric values, `*_delta` tables compare with one stream scoring the same selected answer, and `*_vs_mean` tables compare with arithmetic pooling. The full tables include accuracy, ECE, t-ECE, AUARC, AUROC, Brier, t-Brier, and NLL. Coverage and exploratory dataset-block sign tests are saved separately.

The reference stream is the group member selected by fitting-split initial-answer accuracy; it scores the same final answer as every pooling method. All methods in a group use the same evaluation questions and answer. Accuracy uses all evaluation questions. Deltas are calculated per group before averaging, and the reported `±` is the sample SD across groups. The report does not compare initial answers with post-debate answers or change answer selection.

The t metrics use one output temperature fitted by NLL on fitting questions. The temperature is applied only to evaluation probabilities; raw AUARC and AUROC are unchanged. The paper table shows mean within-group deltas for t-ECE, t-Brier, AUARC, and answer-matched AUARC (all x100), with the sample SD of those deltas. The compact table keeps one CAGE-CAL row; detailed reports retain BetaSB.

Only round-one consistency results are included. The voting judge is not imported; the paired-debate CAGE-CAL scores are joined from their own audited adapter run. The sequence-probability report can be added after its candidate scoring is complete.


The current reporter also joins the paired-debate CAGE-CAL adapter in
`baselines/results/debate_adapter/71259088b2f0e984`. Its raw row has
validation-fitted t-Brier and t-ECE; its BetaSB row has no second temperature.
`atomic.csv` retains AUARC and adds `answer_matched_auarc` and the corresponding
delta. In debate the reference already scores the final answer, so the two
AUARC columns are numerically identical.

To select the reference by fitting-split AUARC instead of fitting accuracy:

```bash
python paper_results/codes/debate_protocol/run.py \
  --run results/debate_composition/run_618a3c11ea4498e0 \
  --reference fit_auarc --out paper_results/results/debate_protocol_fit_auarc
```

Keep the alternative output separate from the primary report.
