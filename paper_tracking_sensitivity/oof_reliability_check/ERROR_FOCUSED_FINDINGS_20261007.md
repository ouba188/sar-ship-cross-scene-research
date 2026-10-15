# Error-focused expert: fixed follow-up also negative

## Decision And Disclosure

Do not promote. This study was declared AFTER the complementary-score
experiment failed, BEFORE its own query readout; it is not a rescue arm
retroactively added to the first protocol. No subsequent weight sweep or
third experiment is queued. Current50.382098% BA main baseline is unchanged.

Output: `E:/Hermes/paper_tracking/sensitivity/oof_error_focused_results_20261007`.
Protocol: `ERROR_FOCUSED_PROTOCOL_20261007.md`.
23 ports, five original64-label supports,230 new expert fits,575 policy rows
and345 full-query expert rows,9.94 seconds. No base/gate/backbone fitting.

## Mechanism

Keep original E0-anchored residual expert and fixedlambda8. Give a support
sample difficulty2 if the adapted base's leave-one-out prediction is wrong,
otherwise1. Normalize difficulty WITHIN its true class so that every class
retains exactly64/K_present total weight. Thus this is error emphasis, not
a changed class prior or larger effective training mass. Compare an otherwise
identical in-sample-B64 error weighting and the old uniform-balanced expert.
LOO uses cached63-label base scores with fold-local weights; expert fitting
still uses all64 labels. The factor2 is fixed, not optimized or claimed novel.

## Budgeted Results (%)

| Policy | BA | Accuracy | Expert admissions |
| --- | ---: | ---: | ---: |
| Current visual handoff | 50.382098 | 56.653819 | 4.932516 |
| LOO-error-weighted expert | 50.290293 | 56.693311 | 4.932516 |
| In-sample-error-weighted expert | 50.294098 | 56.707298 | 4.932516 |

LOO-error primary delta -0.091805 BA points:9 positive/14 negative ports,
descriptive port-bootstrap95% interval[-0.179356,-0.020192], four negative
draw means. Draw differences:
[0.035101,-0.161022,-0.081184,-0.041457,-0.210465]. Accuracy+0.039492 points
does not rescue the declared BA primary. In-sample control also loses
0.088000 points,6 positive/17 negative ports, all five draw means negative.
LOO minus in-sample is -0.003806 points, not a supported LOO contribution.

The new policy beats adapted base alone by1.750223 points, but the old
expert already beats it by1.842028. Its expert-admission count matches the
old policy exactly in every context, not merely the same nominal cap.
Original-source-base changes are29.733547%, NOT bounded by5%.

## Full Expert Quality

Not5% policies: original expert BA52.894867%, LOO-error52.835233%,
in-sample-error52.843986%. LOO-error minus old -0.059634 points,8 positive/
15 negative ports. Accuracy58.653583% versus old58.425765% is a different
metric, not a net BA improvement. LOO-error rescues0.429315 BA points where
both adapted base and old expert were wrong, but recovers0.630865 points
from old expert errors and harms0.690499 points of its correct predictions.

Both the score-residual and this weighting construction create some real
new rescues, but lose at least as much elsewhere. This does not prove all
complementary learning impossible; it does rule out promoting these fixed
implementations or adding them as successful independent paper modules.
Retain the source-anchored strong baseline and these failure records.

## Verification And Limits

Four focused tests pass: per-class mass/exact2:1 difficulty, recovery of the
old fit for class-constant difficulty, independent augmented-SVD weighted
fit and score-magnitude invariance. Producer checks original policies and
metrics, old expert, class masses and actual handoff caps. Independent
artifact replay passed in7.13s:230 augmented-SVD correction refits, all575
policy and345 full-expert rows, weighting signals, per-class masses, receiver
masks and summary contrasts. Maximum coefficient error1.11e-14, score
1.55e-14, class-mass2.13e-14 and metric3.33e-16. See `VERIFICATION.json`.
All115 LOO-error admission counts equal the old policy's counts. Both study
processes and verifiers have finished; no additional run is queued.

Development exposure, row-only target support/query split, uncalibrated
frozen gate for adapted scores and admission-not-compute limitations remain.
No source facility/AIS data was added, no manuscript claim upgraded, and
neither target outcomes nor port identities choose a deployment winner.

User direction at closeout: omit experiments whose sole purpose is warding
off hypothetical criticism. Future work must change a method decision,
explain the observed mechanism, or advance the central argument. Do not add
more controls or verification to these completed negative studies.
