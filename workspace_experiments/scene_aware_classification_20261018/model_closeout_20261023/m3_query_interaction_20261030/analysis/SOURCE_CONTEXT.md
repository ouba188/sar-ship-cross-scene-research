# Reused source coverage and startup context

Coverage is inherited from `m3_pairwise_repair_20261030`, without re-auditing source images or opening source prediction archives:

| Dataset | Source observations | Ships | Pass groups | Classes | Legal class pairs | Legal observation pairs | Actual reference bank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| private | 136 | 128 | 20 | 8 | 28 | 6562 | 16 |
| OSS2 | 241 | 235 | 4 | 4 | 6 | 13757 | 32 |

Sources: previous round `private/pair_coverage.json`, `oss2/pair_coverage.json`, and `FINDINGS_ZH.md` lines 11-13. Legal pairs are combinations sharing observations, not independent observations. The table describes the eligible source pool, not a newly measured count of unique examples actually sampled in this round.

The current experiment reads the previous round `exposures/schedule_s{seed}.json` directly and maps its IDs to the source rows. Per arm, private uses 54 optimizer steps and OSS2 96, batch size 16 and LR 1e-4. Reference counts 16/32 are bank sizes, not numbers of independent cases. Repeated presentations and repeated training orders do not increase the unique ship count. Development queries are private160 and OSS2 validation228 only.

Main reported that startup initially compared four-query inference with the first four rows of a batch-16 inference at tolerance 1e-6. That failed before training. The corrected check compares identical four-query batches and retains tolerance 1e-6; full baseline logits must still equal the previous CE archive exactly. Current experiment code confirms both comparisons and records `initial_error` separately from `baseline_batch_error`. The latter is a numerical batch-context difference, not evidence of a trained-model change. Numerical values will be taken from completed receipts only after READY.

At the private17 READY handoff, all three private arms were reported complete with worker exit 0; public17 was still running. This note records source/startup provenance, not public evaluation results.
