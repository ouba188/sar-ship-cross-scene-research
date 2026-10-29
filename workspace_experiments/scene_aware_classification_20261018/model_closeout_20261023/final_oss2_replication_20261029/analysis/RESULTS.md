# OSS2 four-class numerical results

OSS2 final spatial M12+M3 versus plainR50, explicit FOUR classes mapped from original [0,2,3,4]. New architecture on the historical fold0, NOT a new blind test. Evaluation: 378 queries, 369 ships, five pass groups. Reference bank: 32 actual source-fit identities, eight per class. Plain has NO reference contribution. Full versus plain includes architecture and training-objective differences; no pure-architecture or physical-causality claim. Fit160/adapt96 steps per chain. Seeds17/29/43 change order under shared initialization, NOT three independent initializations. OA/BA are percent; deltas are percentage points; NLL is natural log; Brier sums FOUR classes. BA averages present classes only; absent groups are blank. SD is sample SD across three orders. Paired pass bootstrap resamples five whole passes and averages paired fixed-seed differences; its interval is conditional on these fitted models and only five clusters, not a training-uncertainty CI. Repeated observations, seeds and references are not independent evidence. Validation228/selection194 are recorded available only; no checkpoint selection. Outcome-selected native cases do not estimate population proportions; generated images remain not_reviewed.

| Condition | OA mean / SD | BA mean / SD | NLL mean / SD | Brier mean / SD |
|---|---:|---:|---:|---:|
| full_fit | 62.169312 / 1.153148 | 57.293447 / 1.010582 | 1.288564 / 0.020573 | 0.568360 / 0.010756 |
| plain_fit | 54.938272 / 0.404107 | 53.390851 / 1.219592 | 1.202578 / 0.050936 | 0.619840 / 0.015272 |
| full_adapt | 61.375661 / 2.306296 | 57.662916 / 1.827268 | 1.274771 / 0.018147 | 0.574993 / 0.017636 |
| plain_adapt | 55.643739 / 1.504296 | 53.537833 / 3.377138 | 1.124319 / 0.018379 | 0.603681 / 0.004787 |

| Paired comparison | Metric | Mean delta | Order SD | Pass CI low | Pass CI high |
|---|---|---:|---:|---:|---:|
| full_adapt_minus_plain_adapt | OA | 5.731922 | 1.800756 | 0.377358 | 9.535067 |
| full_adapt_minus_plain_adapt | BA | 4.125083 | 2.640427 | 0.848214 | 6.548557 |
| full_adapt_minus_plain_adapt | NLL | 0.150452 | 0.035534 | 0.049598 | 0.231595 |
| full_adapt_minus_plain_adapt | Brier | -0.028689 | 0.018709 | -0.060788 | 0.033844 |
| full_fit_minus_plain_fit | OA | 7.231041 | 1.331540 | 2.641509 | 11.475410 |
| full_fit_minus_plain_fit | BA | 3.902596 | 0.980053 | -1.864316 | 7.343374 |
| full_fit_minus_plain_fit | NLL | 0.085985 | 0.042621 | 0.010730 | 0.206938 |
| full_fit_minus_plain_fit | Brier | -0.051480 | 0.020658 | -0.084677 | 0.010790 |
| full_adapt_minus_full_fit | OA | -0.793651 | 1.734772 | -3.300330 | 0.910125 |
| full_adapt_minus_full_fit | BA | 0.369469 | 1.067563 | -0.865312 | 1.813244 |
| full_adapt_minus_full_fit | NLL | -0.013792 | 0.032920 | -0.047486 | 0.001135 |
| full_adapt_minus_full_fit | Brier | 0.006632 | 0.014364 | -0.010346 | 0.020148 |
| plain_adapt_minus_plain_fit | OA | 0.705467 | 1.876878 | -1.491681 | 4.213135 |
| plain_adapt_minus_plain_fit | BA | 0.146983 | 2.387617 | -3.690872 | 4.327237 |
| plain_adapt_minus_plain_fit | NLL | -0.078259 | 0.033954 | -0.104670 | -0.025130 |
| plain_adapt_minus_plain_fit | Brier | -0.016159 | 0.013753 | -0.035775 | -0.002591 |

plain_adapt_minus_plain_fit: 0/2000 bootstrap draws omit an originally present class.

full_adapt_minus_plain_adapt, full_fit_minus_plain_fit, and full_adapt_minus_full_fit also have 0/2000 bootstrap draws omitting an originally present class.

| Comparison | State | Seed events | Unique queries | Unique ships |
|---|---|---:|---:|---:|
| full_adapt_minus_plain_adapt | rescued | 132 | 70 | 70 |
| full_adapt_minus_plain_adapt | harmed | 67 | 39 | 39 |
| full_adapt_minus_plain_adapt | CC | 564 | 207 | 203 |
| full_adapt_minus_plain_adapt | WW | 371 | 146 | 145 |
| full_fit_minus_plain_fit | rescued | 152 | 82 | 81 |
| full_fit_minus_plain_fit | harmed | 70 | 45 | 45 |
| full_fit_minus_plain_fit | CC | 553 | 212 | 208 |
| full_fit_minus_plain_fit | WW | 359 | 150 | 149 |
| full_adapt_minus_full_fit | rescued | 30 | 23 | 23 |
| full_adapt_minus_full_fit | harmed | 39 | 28 | 28 |
| full_adapt_minus_full_fit | CC | 666 | 253 | 247 |
| full_adapt_minus_full_fit | WW | 399 | 162 | 161 |
| plain_adapt_minus_plain_fit | rescued | 83 | 45 | 45 |
| plain_adapt_minus_plain_fit | harmed | 75 | 44 | 44 |
| plain_adapt_minus_plain_fit | CC | 548 | 202 | 198 |
| plain_adapt_minus_plain_fit | WW | 428 | 165 | 163 |

8 selected queries / 16 separate native VV/VH panels generated; NOT REVIEWED.
See cases/cases.json for all four endpoints, three orders, actual eligible refs and margins. Plain reference contribution is NONE. Numeric shifts cannot identify SAR physical causes. Main-agent review must add visible facts, model-supported evidence, useful conditions and unresolved gaps.
