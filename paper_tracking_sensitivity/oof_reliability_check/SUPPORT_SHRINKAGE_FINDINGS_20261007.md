# Support-LOO shrinkage: primary negative, secondary gains unstable

## Decision

Do not replace the current adapted-base + expert-handoff baseline. The
prespecified primary (shrink base only, then handoff) loses 0.036595 BA
points. Expert-only shrinkage gains 0.108124 points in the mean, but has
7 positive / 10 negative / 6 unchanged ports, three negative support-draw
means, and lower accuracy. This is not a reliable new gain or an established
innovative module. Preserve all variants, without port-specific mixing or
target-readout parameter scans.

## Setup

Output: `E:/Hermes/paper_tracking/sensitivity/oof_support_shrinkage_results_20261007`.
Protocol: `SUPPORT_SHRINKAGE_PROTOCOL_20261007.md`.
23 ports, five original support draws, same 64 labels shared by both heads.
For each branch, 64 leave-one-out fits use 63 training labels with fold-local
class weights. A balanced squared-error criterion on held-out predictions
chooses alpha in [0,1]. Deployment scales the original all64-fitted residual.
No additional target labels, features, backbone training or source-gate fit.
14,720 small LOO fits, 230 all64 parent refits, 805 policy rows; 12.79 seconds
for the producer. Mean base alpha 0.593396; expert alpha 0.645835.

The changed setting remains explicit: 5% bounds expert handoffs relative to
the actual adapted fallback, NOT all changes from original source predictions.
All receiving scores are computed before selection; no sparse-execution claim.

## Equal-Port Results (%)

| Policy | BA | Accuracy | Expert admissions | Changes from original base |
| --- | ---: | ---: | ---: | ---: |
| Original visual-adapted base | 48.540070 | 54.794330 | 0 | 27.244958 |
| Original visual-adapted base + handoff | 50.382098 | 56.653819 | 4.932516 | 29.692274 |
| Original bias-adapted base + handoff | 49.929602 | 52.635672 | 4.932516 | 23.246426 |
| Shrunk base alone | 48.491713 | 52.867786 | 0 | 17.807295 |
| Shrunk base + original expert handoff (primary) | 50.345504 | 54.995308 | 4.932516 | 21.630677 |
| Original adapted base + shrunk expert handoff | 50.490222 | 56.459776 | 4.922671 | 28.832373 |
| Both branches shrunk + handoff | 50.470808 | 54.829301 | 4.925953 | 21.311848 |

All metrics average the five draws within each port, then average ports.
Admission caps are matched; actual counts differ slightly for shrunk experts.

## Paired Contrasts

Primary versus original visual handoff: -0.036595 BA points, 6 positive /
11 negative / 6 tied ports; descriptive port-bootstrap 95% interval
[-0.534353, 0.449871]. Draw differences are
[0.123497, 1.170039, -0.600767, -0.499189, -0.376553] points.
Accuracy declines 1.658511 points. Primary development screen fails.

Expert-only shrinkage versus original visual handoff: +0.108124 BA points,
interval [-0.289405, 0.694396], accuracy -0.194044 points. Draw differences:
[-0.012686, 0.280196, -0.433887, -0.177678, 0.884675]. Mombasa gains
5.683309 points while Newcastle loses 1.346857 and Los Angeles 1.157917.
This concentration is a diagnostic, not a reason to deploy a port-specific
winner. Six tied ports do not count as positive.

Both shrunk versus original visual handoff: +0.088710 BA points, 6 positive /
11 negative / 6 tied ports; interval [-0.781776, 1.119902]. Accuracy falls
1.824518 points. Mombasa gains 8.634853, but Port Hedland loses 5.065334.
Neither shrinkage treatment solves negative transfer across ports.

Expert handoff still adds 1.853790 points over the shrunk base alone, with
22 positive / 1 negative ports and all five draw means positive. Thus the
previous evidence for useful handoff after base adaptation remains; this
experiment fails to establish an additional reliable shrinkage improvement.

## Interpretation And Next Direction

Scalar shrinkage reduces both changes and harm relative to the original base,
but also loses recoveries. Fewer changes alone are not better selection.
Balanced squared-loss LOO strength is not the same objective as improved
classification or positive benefit within a 5% handoff set. The unchanged
source gate is still not recalibrated for these adapted score distributions.

Keep the 50.382098% workflow as the main research baseline. A future bounded
study should learn complementary corrections for the adapted base's errors,
with source-episode or support-held-out supervision. That is a proposal, not
an implemented or validated module. Do not repeat the already failed fixed
4D source-episode bias-receiver gate under a new name. This run does not
trigger another parameter search or additional experiment queue.

All targets have historical development exposure; support/query exclusion
is row-only, not vessel/product-disjoint. Bootstrap intervals are descriptive,
not independent confirmation. No journal-level novelty claim or manuscript
result is upgraded from these findings.

## Verification

Four focused tests pass: independent augmented-SVD LOO fits with singleton
classes, omitted-label independence of its prediction, numerical scalar-loss
minimization, and zero/harmful/underfit residual boundaries. The producer
reproduces all64 parent scores and all three original control policies and
metrics. Independent artifact verification passed on its first run (7.61s):
all 805 metric rows/class recalls and CSV rows, all six summary contrasts,
230 augmented-SVD all64 coefficient refits and shrunk-score/alpha replays,
575 gate/mask replays, and 345 exact parent controls. All 805 policy masks
and predictions agree. Maximum coefficient error 1.02e-14, gate error
4.60e-14, metric error 3.33e-16. No tolerance changes were needed.

The bounded LOO audit independently fit 40 folds across the first two sorted
ports (Antwerp-Bruges and Busan), all five draws and both branches, rows 0/63.
Maximum LOO error 3.64e-15; each refit recomputed training weights from 63
rows. These audit supports contain no singleton classes; singleton behavior
is covered by the focused synthetic tests, not a claimed real-data singleton
audit. The verifier does not independently refit all 14,720 LOO models.
See the run's `VERIFICATION.json` for the full scope and inherited-provenance
limitations. All experiment and verifier processes finished; no next job is
queued. Manuscripts and the original main-model results remain unchanged.
