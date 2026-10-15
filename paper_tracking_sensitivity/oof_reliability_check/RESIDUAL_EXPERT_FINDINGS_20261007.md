# Source-anchored residual expert: positive development screen

Date: 2026-10-07. Run: ../oof_residual_expert_results_20261007_v2.
Protocols: RESIDUAL_EXPERT_PROTOCOL_20261007.md and separately declared
RESIDUAL_COVERAGE_CONTROL_20261007.md. Original baseline artifacts unchanged.

## Decision

Prioritize source-anchored target visual correction over further gate-feature
stacking or direct facility concatenation. This fixed classifier experiment
passes both prespecified positive-mean criteria. It is a genuine development
gain, not proof of top-journal novelty, independent generalization, or a
completed budgeted handoff method. Keep all baselines and negative ports.

## What changed

The 64 target labels train a small eight-class residual head on frozen visual
features. The original source expert's scores remain the prior. A penalized
correction adjusts its decision boundaries; it need not choose either old
argmax. The source class coding (+1/-1), target class balancing, source-only
feature scaling, and fixed regularization strength 8 are explicit. There is
no backbone training, parameter sweep, extra label or facility input.

This differs from historical M5, which independently trained target64 ridge
and blended/replaced predictions on an old selected 5% subset. It does not
invent residual learning: local historical probability-residual and conditional
log-probability correction experiments already exist under other protocols.

## Same-support results

All 23 active ports, five original uniform support draws per port, exactly
64 labels in each draw. Evaluate all other query rows. Average within port
over draws, then equally across ports. Accuracy below is also equal-port,
not a pooled sample-weighted accuracy.

| Model | BA, % | Accuracy, % |
|---|---:|---:|
| Original fusion expert | 47.0938 | 49.0099 |
| Original four-feature OOF gate | 48.0408 | 50.3226 |
| Historical target64 ridge recipe, same five draws | 43.7883 | 61.6086 |
| Matched target-only ridge, zero prior | 41.3907 | 55.3066 |
| Source-anchored bias correction | 51.7568 | 53.7238 |
| Source-anchored visual residual expert | **52.8949** | **58.4258** |
| Fusion restricted to support-observed classes (follow-up) | 51.0951 | 57.9531 |

The historical target64 recipe still wins accuracy over the new expert by
3.1828 pp; the new expert instead wins BA by 9.1066 pp. Do not claim dominance
on every metric or describe the matched scratch model as the strongest
target-only baseline.

### Prespecified comparisons

| Visual residual minus | BA difference, pp | Positive ports | Descriptive port-bootstrap 95% interval, pp |
|---|---:|---:|---:|
| Original fusion | +5.801050 | 20/23 | [2.508607, 9.453558] |
| Anchored bias only | +1.138092 | 19/23 | [-0.147661, 2.135366] |
| Original OOF gate | +4.854098 | 20/23 | [1.954715, 7.994775] |
| Matched target-only ridge | +11.504186 | 23/23 | [9.068944, 13.955492] |
| Historical target64 recipe | +9.106615 | 23/23 | [6.089124, 12.150737] |

The gain over the original gate is positive in all five draws (4.6485 to
5.0422 pp). The visual-minus-bias gain is negative in one draw (-0.0274 pp)
and its bootstrap interval includes zero. These are overlapping support
draws, not five independent datasets. The mean-based development screen
passes, but the visual component's extra benefit is not yet robustly established.

### Necessary strong control

After the original readout, bias gains motivated ONE separately documented
control: restrict unchanged fusion scores to classes present in the same64
labels. Its policies were all saved before its own readout. This is a
post-readout explanatory follow-up, not part of the original preregistration.

Visual residual minus this control is +1.799763 pp, 15/23 positive ports,
descriptive interval [0.025956, 3.597951] pp. A large part of the original
improvement can be achieved without a visual residual; hard class restriction
does not explain all of this implementation's mean gain. Do not equate
absence from support with absence from the real target population.

### Recoveries and important losses

Relative to fusion, the visual expert recovers 9.6869 BA points and harms
3.8859 points, net +5.8010. Its recovery on rows where BOTH old models were
wrong contributes 7.8089 BA points, weighted over all evaluated classes.
This is gross recovery, not a net improvement or conditional subset accuracy.

Three ports lose to the old OOF gate: Mombasa -12.1825 pp, Newcastle -1.4200,
Callao -0.3579. Mombasa also loses 10.2028 pp against bias-only adaptation.
No port is excluded and no portwise winning model is substituted. In particular,
the aggregate gain does not establish reliable benefit at small target ports.

## Execution and audit boundary

Four focused unit tests passed, including an independent sklearn weighted-ridge
solution, zero residual under an exact prior, source-only feature scaling,
and recovery outside the original model pair. All23-port policies, scores,
support rows and corrections were saved before candidate readout. Original
base/fusion/OOF-gate BA and accuracy match parent results within1e-12.

The complete main run took13.51s:460 small target fits (four recipes across115
contexts),23 source-head reproduction checks,0 backbone/PCA fits. The coverage
control has no training. A first run stopped before readout at a float32
near-tie check; its incomplete files remain in the non-v2 directory.
V2 preserves original prior scores. Reconstructed source heads have maxscore
error1.21e-5 and2 near-tie argmax differences; neither reconstruction replaces
the frozen original prior. See protocol for the numerical-only clarification.

Independent verification completed with exit0: VERIFICATION.json in the v2
run. Separate code replayed805 main policy/metric rows and575 coverage-control
rows, source-only geometry, support/evaluation alignment, score-to-prediction
mapping, confusion-based recall/recovery/harm, and all aggregate contrasts.
Four independent sklearn support fits checked the optimizer; maximum score
replay error8.35e-7. Source-head agreement relies on the producer's reproduction
receipt, not a second head-training audit. No encoder retraining was performed.

All targets remain historically exposed development data. Outer source
identity exclusions are inherited, but target support/query separation is
row-only, not vessel/product-disjoint. This matters especially for a target
visual adapter. Frozen-encoder access is the prior user declaration, not a
newly certified representation-training lineage.

## Next Research Step

Use this as an effective expert candidate, not an already finished routing
paper. The next narrow experiment should reconnect it to the original
budgeted handoff under the same annotation and replacement budgets, retaining
bias-only and hard observed-class controls. Train/select any new routing
criterion on source episodes rather than target query outcomes. Report
replacement cost separately from actual expert feature-computation cost.

The full-query gain reported here must not be relabeled a5% handoff gain.
Small-port harm and the incremental value over simple calibration remain the
mechanism questions to solve. Do not broaden to arbitrary attention modules,
facility concatenation or a post-result regularization search.
