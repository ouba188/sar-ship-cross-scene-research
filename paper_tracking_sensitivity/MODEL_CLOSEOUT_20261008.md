# Model Design Closeout And Next-Round Handoff

Date: 2026-10-08. User approved closing the current round and deferring the
next experiment round. No new training or evaluation is started by this note.

## Subsequent Resume: Joint Pilot Completed

The user subsequently requested continuation. The planned matched2x2 experiment
is now complete in joint_residual_pilot_20261008/JOINT_FINDINGS.md.
JointOA29.758639/BA25.431755 versus matchedreference29.634353/25.106684:
+0.124286ppOA / +0.325071ppBA. Relative to matchedaligned-only, OA+0.432165pp
in all5draws and BA-0.003211pp; relative to complement-only, BA+0.428282pp
in all5draws with OA-0.070650pp. Adopt joint as a balanced F0921 candidate.
The original design/handoff below is retained as historical context; its
"not started" and "no joint checkpoint" statements are now superseded ONLY
for this F0921 pilot. No publicOSS/24port jointvalidation has been completed.
The training operator includes the finalcomplement branch,46tests pass,
all20arm/draw policies retain the shared64labels and5%budget. No main-model
cross-dataset tables or old artifacts were overwritten.

## Current Decision

Freeze the proposed four-module structure for writing and the next integration
experiment. This is a DESIGN closeout, not a claim that the combined model has
been trained, jointly validated or established as a top-journal contribution.
Existing main-model artifacts and separate candidate results remain unchanged.

Working description: deployment-consistent cross-port complementary expert
handoff under a shared target-label budget and a restricted admission budget.

## Four Modules

1. Source-prior-anchored dual-branch adaptation. Paired VV/VH R50 provides the
   base representation; VH SSL R18 supplements the expert. Both classifiers
   adapt around their source priors with the SAME64 target labels and the
   existing class-balanced residual ridge. Do not count the shared labels twice.
2. Projection-complement information retention. Keep the source-PCA branch and
   expose the separately computed R50/R18 PCA-complement coordinates to expert
   adaptation. Current tested recipe: .5K_retained + .5K_complement +1, with
   source-only normalization, target residual ridge8 and one penalized bias.
3. Deployment-aligned expert representation training. Source-port64-support
   episodes optimize query classification through the actual residual solver.
   Update R18 layer4, retain R50 and source-fitted head/projection parameters.
   The combined version MUST include module2 in this training forward pass.
4. Receiver-consistent budgeted handoff. Existing source-OOF reliability gate
   reads the actual adapted base/expert outputs. Disagreement, observed-class,
   positive-score eligibility and ranked floor(5%*query_count) admission remain.

Module3 is a training procedure, not another inference network. Modules1/2
interact in the expert correction; the four modules are not four independent
sequential networks or four independently established original algorithms.

The5% limit covers expert admissions, not global base adaptation or measured
feature-computation savings. Port knowledge currently means learned source
statistics/classification/reliability, not facilities, berth roles or AIS tracks.

## Verified Candidates And Limits

Stable main framework on23eligible target-port tasks: equal-port base64
BA48.540070%, full5 BA50.382098%; handoff increment1.842028pp,21/23ports positive.

Deployment-aligned residual_global on correctedF0921: fullOA29.326474%,
BA25.435555%. Versus its matched frozen replay: -.307879ppOA / +.328871ppBA.
This is a BA-oriented candidate, not a uniform improvement.

Projection-complement full-model results and deltas versus its matched reference:

| Dataset | OA | BA | Delta OA pp | Delta BA pp |
| --- | ---: | ---: | ---: | ---: |
| F0921 | 29.828803 | 25.003274 | +0.195260 | -0.101876 |
| Public OSS1 | 43.284047 | 31.970674 | +0.217899 | -0.359902 |
| Public OSS2 | 32.442271 | 29.971866 | +0.059766 | +0.033502 |

Complement expert all-queryOA improves in5/5draws per dataset. Its full-model
OA gain is usable under the user's adoption policy, with negative BA retained.
The complement has not yet been evaluated across24ports. The two candidates
have NOT been jointly trained/evaluated; their gains cannot be added.

Do not include failed spatial attention, source-supervised8Dcomplement
compression, or the extra handoff CE term. Half-retained regularization has a
separate F0921 BA gain but is not a new information module. Input alignment
repairs remain preprocessing corrections, not another claimed innovation.

## Next Round, Not Started

First run a bounded matched2x2 integration experiment on correctedF0921:

| Arm | Complement | Deployment-aligned training |
| --- | --- | --- |
| Existing reference | Off | Off |
| Complement only | On | Off |
| Aligned only | Off | On |
| Joint candidate | On | On |

For the joint candidate, differentiate through the complement-enabled final
expert. Simply attaching complement features to the existing PCA-only-trained
checkpoint does not establish training/deployment alignment. Frozen source
HEAD PARAMETERS do not imply frozen score values when encoder features change.

Retain existing source identity exclusions,64target support draws, base64
predictions, ridge strength, gate policy and5%cap. Match neural initialization
and training recipe between the aligned-only and joint arms. Fix the combined
recipe before target readout; do not choose per-dataset winning configurations
and describe them as one universal model.

Read out OA and BA, expert-level and final-policy gains, and paired draw effects.
Report a useful metric-specific gain without requiring every draw or metric to
improve. If the joint version does not add useful value, keep the separately
supported configuration and close the branch. Expansion to publicOSS/24ports
follows only when needed to substantiate the selected final model. Do not reuse
a neural checkpoint across datasets without respecting its source-cohort scope.

## Evidence And Entry Points

- oof_reliability_check/BASE_ADAPT_FINDINGS_20261007.md
- aligned_residual_pilot_20261008/ALIGNED_FINDINGS.md
- projection_residual_pilot_20261008/PROJECTION_FINDINGS.md
- source_discriminant_pilot_20261008/SOURCE_DISCRIMINANT_FINDINGS.md
- information_loss_diagnosis_20261008/INFORMATION_LOSS_FINDINGS.md
- CANDIDATE_MODULES.md
- oof_reliability_check/train_aligned_residual.py
- oof_reliability_check/aligned_residual_ops.py
- oof_reliability_check/projection_residual_expert.py

The existing12-test complement suite passed; prior aligned-operator tests and
run-specific replay checks are recorded in their reports. No combined-model
test result or checkpoint exists yet. Historical dataset/encoder exposure and
row-level target support/query limitations remain unchanged.
