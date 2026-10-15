# Source-Prior-Preserving Polar Evidence

## Decision

The requested new-information module has been implemented and tested, but
neither spatial-attention version establishes an incremental accuracy benefit.
Retain aligned-global as the current BA-oriented candidate. Stop this bounded
polar-attention branch here rather than choosing target-driven loss weights.
Keep the code, checkpoints, positive reference comparisons, and failed branches.

## Structural Follow-Up

The preceding unconstrained paired-information run changed the expert's
inputs but degraded its full-query recognition. This exploratory follow-up
therefore preserves the original e206 encoder AND source prior. Only the
new residual projection and attention parameters train.

For learned512D polar residual delta, source PCA rows P,64D expert scales s,
and source classifier SSL block W(8x64):

    dz = (delta @ P.T) / s
    N = I - pinv(W) @ W
    dz_new = dz @ N
    delta_new = (dz_new * s) @ P
    feature = original_VH_feature + delta_new

The added feature lies in the source classifier's null space but remains in
the retained PCA space. Thus it can alter the support/query residual kernel
and the64-label adapted prediction without intentionally changing the
unadapted source class scores. It is not a zero-output module. Both source
and target inference use their own fixed source-fit geometry and the same
projection formula. The target encoder still consumes no target labels.

The original e206 layer4 is frozen; both pooled and spatial-contrast arms
train new parameters for6epochs atlr1e-3, same20source episodes,28402source
queries,64supports per episode,128query batches, AdamW1e-4weight decay,
clip5,seed20261008. All target supports/queries, base predictions and5%caps
match preceding runs. No encoder/head/PCA/gate refit occurs at deployment.
The source-only training still differentiates through actual ridge8 adaptation.

This follow-up was designed AFTER seeing the first paired pilot; it is
explicitly exploratory, not a preregistered independent confirmation. The
comparison to the preceding pilot changes both the projection and whether
the encoder trains, so it does not isolate a single causal failure source.

## Results

Percentages, means over five overlapping support draws:

| Configuration | Full OA | Full BA |
| --- | ---: | ---: |
| Original frozen replay | 29.634353 | 25.106684 |
| Existing aligned-global candidate | 29.326474 | 25.435555 |
| First unconstrained paired pooling | 28.958639 | 25.355954 |
| First unconstrained spatial contrast | 28.853636 | 25.309197 |
| Prior-preserving paired pooling | 29.432125 | 25.078388 |
| Prior-preserving spatial contrast | 28.980677 | 24.873997 |
| Same preserved contrast model, spatial term off | 29.564351 | 25.113472 |

Prior-preserving contrast minus matched pooling: -0.451448ppOA/-0.204390ppBA.
Contrast minus its spatial-term-off inference: -0.583674ppOA/-0.239475ppBA.
Contrast minus aligned-global: -0.345797ppOA/-0.561558ppBA.
The paired-pooling arm gains0.105651ppOA over aligned-global, but loses
0.357168ppBA and is worse on both metrics than the original frozen model.
Record that positive OA comparison without promoting a dominated variant.

Full-query expert diagnostics, NOT within the5%deployment budget:
pooled29.419161%OA/24.072798%BA, contrast28.105489%OA/23.251380%BA,
spatial-term-off30.627993%OA/24.814750%BA. As in the first pilot, the
current negative attention result is not solely a handoff-gate bottleneck.

## What Is Established

- Actual new expert-branch information: pairedVV/VH spatial feature maps,
  previously VH-only in that branch. The overall base already used both.
- Actual trainable extraction: retained source competitor directions guide
  two spatial attention maps over polarization differences; gradients pass
  through the support-adapted prediction into the new module.
- No discarded auxiliary classifier. Source labels train the module; exactly
  the existing64target labels adapt the deployed residual expert.
- Mathematical prior preservation, with finite-precision readout explicitly
  measured against the original frozen expert on all123489target rows:
  max score differences4.47e-6(pooled),1.323e-5(contrast),5.54e-6(termoff);
  source-prior argmax changes0,1,1 respectively. Do NOT claim bit-exact or
  zero-changes for all three arms. Saved in PRIOR_PRESERVATION.json.
- The learned contrast remains active: mean contrast norm5.750653, actual
  feature change when disabling it17.936578. Activation does not imply gain.
- Two projector tests pass: source-score preservation while adapted scores
  change, and gradients through both support/query evidence. Together with
  the previous24module tests,26new focused tests passed. Both complete
  training/evaluation pipelines exit successfully and metric counts reconcile.

What is NOT established: beneficial spatial contrast, broadly improved
cross-dataset recognition, a novel calibrated polarimetric quantity, or a
top-journal-ready innovation. Preserving old class scores alone does not
guarantee useful support-adaptation geometry. This does not prove that all
dual-polarization or spatial-attention methods fail.

## Artifacts

The first module/design/failed result remain in ../polar_residual_pilot_20261008.
This folder holds the follow-up protocol, retained source model, checkpoints,
features, policies, metrics, summaries, diagnostics and prior-preservation
readout. Code: support_visible_polar.py, test_support_visible_polar.py,
prepare_support_visible_polar.py, with shared train_polar_residual.py and
evaluate_polar_residual.py in sensitivity/oof_reliability_check.

Historical F0921/unlabeled e206 exposure and row-level support/query remain;
five support draws are not five independent datasets or training seeds.
Existing positive candidates remain in ../CANDIDATE_MODULES.md.
