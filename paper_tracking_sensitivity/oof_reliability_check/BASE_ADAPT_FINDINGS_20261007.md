# Adapt the base, then hand off: positive development evidence in a NEW setting

## Decision and boundary first

The same64 target labels can improve the VV-only base. The unchanged visual
expert then adds positive mean benefit under a5% admission cap. This is a
useful strong-baseline result and a candidate research direction, not an
established original algorithm. Do NOT overwrite the earlier fixed-base
results or claim that only5% of original predictions changed.

In this study the base is allowed to adapt GLOBALLY. The final visual-handoff
policy changes29.692274% of original source-base predictions on average,
while expert admissions are4.932516%. Base-only visual adaptation changes
27.244958% with zero expert admissions. The cap constrains expert handoffs,
not all changes from the old source model. The user explicitly accepted this
paper setting on2026-10-07 after the initial readout:5% limits expert handoffs.
Preserve the historical distinction. No measured computation/latency reduction
is established, and the approval is not independent confirmation of a result.

Output `../oof_base_adapt_results_20261007`, protocol
`BASE_ADAPT_PROTOCOL_20261007.md`.23 ports,5 draws,64 UNIQUE support labels
shared by base and expert (not128),230 base fits,805 metrics,8.27seconds.
The expert stays the archived anchor_visual64, with no new expert/backbone
training. All seven policies saved before candidate query-label readout.

## Equal-port results (%)

| Policy | BA | Accuracy | Expert admissions | Changed from original base |
| --- | ---: | ---: | ---: | ---: |
| Original source base | 43.268995 | 44.094334 | 0 | 0 |
| Bias-adapted base only | 48.138134 | 50.105884 | 0 | 20.018647 |
| Visual-adapted base only | 48.540070 | 54.794330 | 0 | 27.244958 |
| Bias-adapted base + handoff | 49.929602 | 52.635672 | 4.932516 | 23.246426 |
| Visual-adapted base + handoff | 50.382098 | 56.653819 | 4.932516 | 29.692274 |
| Old fixed-base handoff | 45.846390 | 46.926903 | 4.932516 | 4.932516 |
| Old action-advantage reference | 45.885986 | 46.899883 | 4.932516 | 4.932516 |

The old/new workflow differences are not matched-total-change comparisons.
They share64 labels and expert admission caps, but differ in whether the base
can adapt. All accuracy values are equal-port, not pooled sample accuracy.

## What is genuinely supported

Visual-base adaptation adds5.271075 BA points over original base:18/23 ports
positive, all five draw means positive, descriptive port-bootstrap95% interval
[1.752046,9.127372]pp. Bias-only already adds4.869139pp. Visual-minus-bias
is just0.401936pp,13positive/10negative, interval[-0.251548,1.015175]. Do not
attribute all adaptation gain to visual boundary learning.

Within the NEW adapted-base setting, expert handoff adds1.842028pp over
visual base alone:21/23 ports positive, interval[1.303648,2.397174]pp. All
five draw means are positive (1.564771..2.165693pp). Accuracy adds1.859489pp.
Thus expert use has incremental value after base adaptation in this study.
Bias-base handoff also adds1.791468pp,22/23 positive. These comparisons, not
just the large old/new overall difference, support keeping the direction.

Visual-handoff minus bias-handoff is0.452496pp,17/23 positive but interval
[-0.237555,1.094063]pp and one draw negative. It does not yet prove a robust
incremental visual-adaptation module beyond bias adaptation.

Versus old advantage reference, the new workflow adds4.496112pp BA and
9.753937pp accuracy,18/23 positive. This is explicitly a changed-setting
development comparison, not a4.5pp novel-gate gain under identical rules.

## Class-support follow-up

Separately declared POST-READOUT in `BASE_COVERAGE_CONTROL_20261007.md`:
restrict original base logits to classes observed among the same64 labels,
without fitting any scores or handing off any query. Output subdirectory
`coverage_control`; all control predictions froze before their own readout.
This is a new explanatory control, not an original preregistered arm.

Its BA47.243422%, accuracy53.224859%, original-prediction changes16.889173%.
Visual-adapted base exceeds it by1.296648pp BA (16positive/7negative;
interval[-0.266008,2.880444]pp, all five draw means positive). Visual-handoff
exceeds it by3.138676pp BA and3.428961pp accuracy,20positive/3negative;
interval[1.816877,4.501327]pp with all five draw means positive.
Thus the mean workflow gain is not fully explained by this hard class
restriction, but much of the original-base adaptation gain is explained by
learning the support's class coverage. Unobserved classes are not assumed
truly absent, and global changes remain permitted only in the new setting.

## Negative ports remain important

Relative to visual base alone, handoff hurts Mombasa0.800262pp and Callao
0.307851pp. Relative to the OLD advantage workflow, new visual-handoff loses
13.612986pp at Mombasa,6.665250pp at Newcastle and2.259593pp at Port Said.
Large gains include Busan23.387654pp and Fujairah24.603819pp. Aggregate gain
does not establish safe transfer to every port. No port-specific winning
policy was substituted, and these outcomes do not select a new gate.

## Implementation and checks

Use exactly the existing source-anchored +/-1 residual recipe, class-balanced
support weights, strength8 and penalized intercept. Bias base updates only
the intercept; visual base uses source-standardized VV128/sqrt128 plus1.
The frozen gate sees the actual adapted base and unchanged receiving expert.
Its margin/entropy distribution is not newly source-OOF calibrated for this
base. Positive-score/disagreement/observed-class guards and integer caps hold.

During fitting, all230 coefficients reproduce sklearn weighted Ridge,
maximum absolute error1.17e-15. The initial receipt called these "primal"
checks; independent review corrected that label because sklearn cholesky
uses a dual solve for129 features and64 samples. No coefficients, policies,
metrics or tolerances changed. Ten augmented-SVD fits on one fixed port
provide an independently formulated numerical check, error1.73e-15.

Independent verification passed: all23 source geometries,230 coefficient/
score replays,230 adapted gate/masks,115 original gate/masks,115 archived
advantage masks and all805 seven-arm metrics. Maximum score error0, gate
error5.78e-15. The output `VERIFICATION.json` explicitly confirms shared64
labels and5% relative to each actual fallback, NOT original prediction changes.

The separately declared coverage control also passed independent replay:
115 control predictions/metric rows,460 parent-comparator metric replays and
four summary contrasts. Its own `coverage_control/VERIFICATION.json` records
the post-readout status and absence of score fitting or expert admissions.

All targets have development exposure; support/query is row-disjoint only,
not vessel/product-disjoint. Encoder/source provenance remains inherited.
The useful next question is adapting the base and expert with explicit
complementarity under a clearly defined label/admission budget. Existing
residual Ridge and fixed confidence gates are strong baselines for that
question, not four newly established top-journal modules.
