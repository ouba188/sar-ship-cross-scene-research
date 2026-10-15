# Usable Development Candidates

## Adoption Policy, Updated 2026-10-08

User decision: fix actual training/deployment mismatches; do not discard a
useful measured gain merely because every metric does not improve. Preserve
positive variants as usable candidates for their stated objective, report
tradeoffs, and distinguish development usefulness from broad novelty claims.
No requirement for all datasets, all support draws, OA and BA to rise together.
Do not spend time on extra experiments whose only purpose is defensive review.
Retain negative results and exact comparisons; do not change what was measured.

## Joint Complement And Deployment-Alignment, Balanced Candidate

Status: preferred OA/BA-balanced F0921 development candidate after matched2x2
integration; not yet a cross-dataset main-model replacement.
Artifact: joint_residual_pilot_20261008/JOINT_FINDINGS.md.

- Four agreed functional modules now run together on correctedF0921. Joint
  training differentiates through the actual final complement-enabled ridge.
- FullOA29.758639/BA25.431755 versus matchedreference29.634353/25.106684:
  +.124286ppOA/+.325071ppBA, eachpositive3/5draws.
- Versus matchedaligned-only: +.432165ppOA(5/5), -.003211ppBA(3/5positive).
  BA is numerically nearlyunchanged, not formalnoninferiority.
- Versus complement-only: -.070650ppOA, +.428282ppBA(5/5positive).
- Joint expertall-queryOA31.874580/BA27.147474, bothmeans abovefrozenreference.
- Same64labels, unchangedbase64, same6171/123425admissions, fixedsourcehead
  parameters/PCA/normalization/gate; no posttrainingrefit ornewtargetlabels.
-46 CPU/CUDA tests pass; actualepisodeoperator parity<3.72e-7. Allfour policies
  savedbefore queryreadout. One neuralseed, historicaldata, noall-drawguarantee.
- Sourceprior SCORES changewith learnedfeatures althoughhead PARAMETERS arefixed.
  Jointresult isnot post-hoc attachment to the PCA-only-trainedcheckpoint.
- PublicOSS/24port jointvalidation notrun. Keep independentcandidate results
  separate; no furtherattention/OT/source8Dcompression/gate branches added.

## Projection-Complement Expert, OA-Oriented

Status: usable OA-oriented development candidate; not the BA-primary model.
Artifact: projection_residual_pilot_20261008/PROJECTION_FINDINGS.md.

- Preserve source prior and existingPCA branch; expose sourcePCA-complement
  coordinates to the same64support residual ridge via fixedhalf/half kernel.
- FullF0921 OA29.828803/BA25.003274: +.195260ppOA/-.101876ppBA vsoriginal.
- FullpublicOSS1 OA43.284047/BA31.970674: +.217899ppOA/-.359902ppBA.
- FullpublicOSS2 OA32.442271/BA29.971866: +.059766ppOA/+.033502ppBA.
- Expert all-queryOA improves1.754264/2.319066/1.814724pp respectively,
  positive in5/5draws on eachdataset. Expert BA isnot uniformly improved.
- Noextra labels, backbone training, gate changes, AIS or geometrymissingness.
  Same64labels and5%handoff; referencepredictions/admissions exact inall15draws.
- Half-retained control separates informationaddition from oldbranchstrength.
  Complement improvesOA vscontrol onall3; onlyOSS2 haspositivefullBA contrast.
- Preserve simplehalf-retained asF0921 BA-oriented regularization alternative:
  OA29.573911/BA25.303587, +.198438ppBA(4/5) and-.059631ppOA vsoriginal.
  Thisisnot a newinformationmodule; aligned-globalstillhas higherBA25.435555.
- Source-supervised8Dcomplement follow-up losesbothmeans onallthree; disabled.
  Report: source_discriminant_pilot_20261008/SOURCE_DISCRIMINANT_FINDINGS.md.
- No24port module evaluation yet; no universal novelty/SOTA/significance claim.

## Utility-Tail Expert Representation

Status: usable BA-oriented development candidate on corrected F0921.
Artifact: neural_complement_pilot_20261007/PILOT_FINDINGS.md.

- Full5 BA25.319728%, OA29.565971%.
- Versus matched frozen replay: +0.213044pp BA, -0.068382pp OA.
- Versus ordinary supervised layer4: +0.322151pp BA, +0.318412pp OA;
  both metrics positive in all five support draws.
- No additional target labels; same64labels and5%handoff.
- Actual layer4 representation learning, not only reweighting frozen features.
- Auxiliary neural classifier is discarded at deployment. That known head
  mismatch motivates the aligned-residual successor; do not hide it by naming.

This entry records the observed gain and its usable scope, not a claim that
the configuration uniformly improves recognition or is already journal-ready.

## Deployment-Aligned Residual Expert Training

Status: preferred BA-oriented candidate among the current F0921 neural pilots;
old frozen model retained as the higher-OA alternative, not overwritten.
Artifact: aligned_residual_pilot_20261008/ALIGNED_FINDINGS.md.

- Use residual_global: actual64support ridge adaptation lies in the training
  gradient path. Retain source PCA/prior/gate at deployment, no head swap.
- Full5 BA25.435555%, OA29.326474%.
- Versus frozen replay: +0.328871ppBA(4/5draws), -0.307879ppOA(0/5positive).
- Versus previous utility-tail: +0.115827ppBA, -0.239498ppOA.
- Versus unchanged Base64: +0.533768ppBA, +0.389386ppOA.
- Same64target labels and5%handoff. No target-label encoder training.
- Extra handoff-focused CE is not enabled: it lowers both mean metrics versus
  the matched aligned-global arm. Record that negative branch, do not count
  it as another independently effective module.
- The prediction-operator mismatch is fixed; hard selection still uses a
  detached minibatch approximation. This is a development result, not a
  fully optimized differentiable deployment-budget objective or novelty proof.

## Paired Spatial Information Prototypes

Status: implemented and evaluated; no incremental spatial-attention benefit
established. Do not replace aligned-global with these prototypes.
Reports: polar_residual_pilot_20261008/POLAR_FINDINGS.md and
support_visible_polar_20261008/SUPPORT_VISIBLE_FINDINGS.md.

- Real expert input change: VH-only to pairedVV/VH16spatial tokens, with
  competitor directions from the retained source ridge classifier.
- Unconstrained poolingBA25.355954/OA28.958639, contrastBA25.309197/OA28.853636.
  Pooling's +0.249269ppBA versus old frozen replay is recorded, but both
  metrics are below aligned-global. Attention loses to matched pooling.
- Prior-preserving residual learns in source-classifier-null directions
  visible to64support adaptation. ContrastBA24.873997/OA28.980677, again
  below both matched pooling and existing aligned-global.
- Preserved poolingOA29.432125 exceeds aligned-global by0.105651pp, but
  BA25.078388 falls0.357168pp and the original frozen model is better on
  both metrics. Preserve the comparison without calling it a useful upgrade.
- Changed information and active attention are verified; positive accuracy
  contribution of the spatial term is not. This is not a threshold/significance
  rejection: the relevant matched mean effects are negative.
