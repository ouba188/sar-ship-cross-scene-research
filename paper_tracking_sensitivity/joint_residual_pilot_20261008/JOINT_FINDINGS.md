# Matched Complement And Deployment-Alignment Integration

Completed 2026-10-08. Decision: retain the joint version as the preferred
OA/BA-balanced F0921 development candidate. Keep the simpler reference and
single-module variants; no cross-dataset main result is overwritten.

## Main Result

All values are mean percentages across the same five64-label support draws.
The R50 base64 predictions are identical in every arm. Exactly6171 of123425
query rows are handed off in each draw; no additional target annotations.

| Arm | Complement | Aligned training | Full OA | Full BA | Expert OA | Expert BA |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Reference | Off | Off | 29.634353 | 25.106684 | 31.055945 | 25.982723 |
| Complement only | On | Off | 29.829289 | 25.003473 | 32.808102 | 25.834198 |
| Aligned only | Off | On | 29.326474 | 25.434966 | 29.806441 | 27.041201 |
| Joint | On | On | **29.758639** | **25.431755** | 31.874580 | 27.147474 |

Expert metrics evaluate the expert on ALL query rows for diagnosis; they are
not deployment performance under the5%budget.

| Comparison | Delta OA pp | OA-positive draws | Delta BA pp | BA-positive draws |
| --- | ---: | ---: | ---: | ---: |
| Joint - reference | +0.124286 | 3/5 | +0.325071 | 3/5 |
| Joint - aligned only | +0.432165 | 5/5 | -0.003211 | 3/5 |
| Joint - complement only | -0.070650 | 2/5 | +0.428282 | 5/5 |

The joint model improves BOTH mean metrics over the matched reference. It
recovers much of the OA lost by aligned-only training, while BA is numerically
almost unchanged (slightly lower by0.003211pp). Relative to complement-only,
it improves BA in every draw at a smaller mean OA cost. This is a useful
tradeoff under the user's adoption policy, not dominance over every variant.
Do not describe numeric closeness as a formal noninferiority result.

The descriptive factorial interaction, joint-aligned-complement+reference,
is +0.237229ppOA / +0.100001ppBA. It supports retaining the combined candidate
in this development study; it is not a significance or universal-synergy claim.

## What Was Actually Integrated

The full candidate now implements the four agreed functional modules together:

1. Source-prior-anchored base/expert adaptation with shared64 target labels.
2. Retained-PCA plus PCA-complement expert information.
3. Representation training through the FINAL complement-enabled residual solver.
4. Existing receiving-expert-consistent gate with the fixed5%handoff cap.

For each source episode, R50 PCA128 and R18 PCA64 are retained. The raw R50
and R18 features also contribute their complements to those source subspaces.
Complement means/scales are fitted only on that episode's identity-purged
source fitting rows, before neural training. R50 complement coordinates are
cached because R50 is frozen. R18 complement coordinates are recomputed in
the differentiable forward pass as layer4 changes.

The joint expert's feature map is:

    phi = [retained192/sqrt(2), normalized_complement2560/sqrt(2), 1]

Each bias-free branch already includes its own division by sqrt(dimension).
Thus its kernel is .5K_retained + .5K_complement +1. One penalized bias is
included. The same support-balanced64x64 ridge8 solve and +/-1 residual targets
are used during source episodic training and target deployment.

Source PCA, normalization and head PARAMETERS remain fixed. Source prior
score VALUES change when the learned R18 representation changes. This is
intentional and occurs in both the training gradient and deployment. The gate
parameters and base64 scores are unchanged, with no post-training head refit.

## Matched Training And Controls

-20 source-port episodes,28402query rows per epoch, same64supports per episode.
- Both trained arms restart from the same original e206 layer4 checkpoint.
- Same episode/query ordering, seed20261008, query batch128, sixepochs,
  AdamW lr1e-5/weight_decay1e-4, gradientclip5, frozen BN running statistics.
- Source-class-balanced adapted-query CE only; no extra handoff CE.
- Gradients pass through support/query R18 features, source prior scores and
  the actual ridge solve. The joint branch is NOT attached only after training.
- Last epoch only; no target checkpoint or mixture-weight search.
- R18 layer3 prefix caches are reused; only layer4 is trained. No target labels
  train the encoder. All four policies finish before this round's query readout.

The two trained arms were rerun together for a matched comparison. Neural
floating-point execution gives a tiny difference from the earlier aligned
pilot (oldBA25.435555 vs current25.434966); use THIS round's matched values.
The untrained controls use the original neural frozen-replay feature/model
cache, so do not splice in the separate earlier raw-feature pilot's slightly
different numerical baseline (OA29.633543/BA25.105149).

## Why The Combination Is Useful

Joint expert OA/BA exceed the reference by0.818635/1.164750pp. Against the
aligned-only expert, joint expert OA improves2.068139pp (5/5draws), with
BA+0.106272pp. Thus the added information has value inside the receiving expert,
not just through different gate rankings.

Mean full-model rescues/harms relative to the unchanged base64:

| Arm | Rescued | Harmed | Net correct changes |
| --- | ---: | ---: | ---: |
| Reference | 1798.2 | 937.6 | 860.6 |
| Complement only | 1989.6 | 888.4 | 1101.2 |
| Aligned only | 1684.2 | 1203.6 | 480.6 |
| Joint | 2068.4 | 1054.4 | 1014.0 |

Joint versus aligned-only adds384.2rescues and removes149.2harms on average.
Versus the original reference, joint adds more rescues but also more harms.
Do not describe the joint system as universally safer or harmful-change-free.

Class recall changes remain mixed. Joint versus reference improves fishing,
general cargo, chemical/product tanker, container and crude-oil tanker mean
recall; bulk, tug and offshore supply decline. Full per-draw recalls are in
metrics.json. The method does not solve all SAR class ambiguity or input loss.

## Verification

-46 CPU/CUDA tests passed for joint geometry and existing aligned operators,
  including explicit NumPy maps, half-kernel weighting, one bias, source-fixed
  buffers and finite-difference support/query gradients.
- Real source64support/32query operator parity errors: aligned3.716e-7,
  joint3.564e-7 against the NumPy deployment residual solver.
- Kernel reference reproduces all archived frozen-replay final predictions
  and admission masks exactly; maximum expert score difference1.035e-6.
- All20arm/draw combinations retain source/target row alignment, support/query
  masks, original64supports, exact base64 predictions and admission caps.
- Both sixepoch training jobs, feature extraction and final evaluation finished.

One GPU entry import initially pulled in an unused legacy OT dependency. The
training script now imports WS/write_json from extract_opensarship2_transfer,
matching the existing GPU trainer, instead of the heavy CPU transfer runner.
No package installation, model change or repeated partial training was needed.

## Adoption Boundary And Reuse

Adopt joint as a balanced F0921 candidate and stop adding modules in this round.
The stable main framework remains the cross-dataset reference until this exact
joint configuration is evaluated elsewhere. The joint checkpoint has NOT been
validated on publicOSS1/OSS2 or24ports. Those separate complement-only results
must not be relabeled as joint-model results.

One neural seed, five overlapping support draws, historical F0921 development
data and inherited target-unlabeled SSL exposure. Support/query is row-level,
not vessel-disjoint. Source OOF is conditional on the learned encoder, not
encoder-level held-out validation. No blind/SOTA/top-journal novelty claim.
The5%budget counts admissions; expert features are computed before selection.

Files: protocol.json; EPISODES_COMPLETE.json; TRAINING_COMPLETE.json;
POLICIES_SAVED.json; metrics.json/csv; summary.json; per-arm policies.npz.
Joint checkpoint: joint/layer4.pt; fixed model: source_model.pkl;
fixed complement statistics: deployment_complement.npz.

Scripts in oof_reliability_check: prepare_joint_residual.py (ClearSAR),
train_joint_residual.py (CUDA environment), evaluate_joint_residual.py
(ClearSAR), joint_residual_ops.py, test_joint_residual_ops.py.
