# Independent Readout Review

2026-11-18. Static code review plus the main agent's saved smoke.json. No model execution, optimizer steps or GPU work by this reviewer. Only analysis.py and this file are reviewer-owned.

## Conclusion

No remaining execution blocker found after the explicit geometry-module import fixed the missing raw_current exports. Native layer1 evidence is genuinely read at 40x40. The live option reads M1-updated layer2 spatial features, not a CCREL summary. The matcher, twenty slots, old mass, and 10x10 write remain unchanged.

Two interpretation limits matter: fine_anchor also changes the layer2 interpolation/pooling functional; fine_live also replaces the protected anchor conv3/BN with the current conv3/BN. Neither comparison alone isolates only the factor suggested by its short arm name.

## Interfaces and Geometry

- model.py:28-62 uses the updated Inputs.rows after polarization recovery. NativeReads calls task.data.prepare, builds native160 tokens.slots from the prepared geometry and footprint, transforms that native mask, then integrates stride-four cells. It does not upsample read20 or rotate read40. Missing/invalid polarizations remain zero. The cache is bounded and contains masks, not detached features.
- sar_m123_current_20261031/sar_m123/input.py:79-87 shifts geometry by crop/pad placement. Prepared geometry is the correct coordinate source, not the unshifted ship_axis center. Full training Inputs.data follows the same transform order at ports24_sync_20261112/train.py:102-124, including recovered rows through pcea_ports24_startpoint_20261117/model.py:28-44 and ports24_recovered_training_20261113/train.py:67-83.
- resolution_matching_20261031/geometry.py:22-30 and tokens.py:46-58 use origin-zero clipped Voronoi cells. Read weights are occupancy fractions. Per-slot native area is sum(read40 * cell_area40), not sum(read40). Descriptor normalization must use its own read-grid sum; keeping matcher mass20 is a deliberate controlled choice, not native-area equality across grids.
- model.py:100-104 uses the historical fine_fixed helper. geometry.py:134-144 leaves f1 native, but interpolates f2 to 40 and pools it with read40. That is not generally identical to pooling f2 directly with read20. The smallest future isolation would pool the two levels on their own native grids, or add a fixed-weight diagnostic retaining old f2 descriptors. This is not requested as an extra training arm and does not invalidate the current combined-readout candidate.

## M1, Hooks and Gradients

- model.py:89-99 repeats M1 and the last layer2 projection to construct descriptors. Inherited reference() repeats the same calculation for PCEA source summaries; staged forward repeats it for the query main path. This is extra compute, not repeated optimizer updates or a second auxiliary loss.
- m1_covariance_20261101/operators.py:6-69 has no BatchNorm, Dropout or in-place input mutation. The inherited full_finetune_20261111/train.py:144-148 puts every BatchNorm2d in eval even during train(). BN affine parameters may still receive gradients in the live path, intentionally; running statistics do not update.
- pcea_ports24_startpoint_20261117/model.py:77-79 only captures M1 responses when context is non-None. Extra extract/reference calls occur outside finish(); pcea_joint_full_20261117/model.py:109-117 creates and clears context around the actual query forward. No extra CCREL/PCEA call is introduced by extract, and old early_x is retained for the main path. No hook contamination found for this supported sequence.
- Query and reference use the same extract implementation. full_finetune_20261111/train.py:172-185 recomputes differentiable references for each training microbatch. No detach or stale-feature bank was introduced. The extra M1 gradient from M3 is intended.
- fine_live minus fine_anchor changes both M1 feedback inclusion and anchor-to-current conv3/BN. Call it live-versus-anchor evidence, not an isolated proof that M1 feedback caused a gain.

## Saved Checks and Remaining Gaps

smoke.json reports legacy replay 0, both M3-off errors 0, unchanged mass/write, fine-grid response 1.109924 to a perturbation invisible to the coarse downsampler (1.28e-16), and direct M1 descriptor gradient 0.062118 for fine_live versus zero for fine_anchor. All recorded trainable groups have finite nonzero gradients. These establish computational access, not classification benefit.

The eight-transform check only tests total native-area preservation on one observation. It cannot detect a rotated or displaced mask with the same area. A focused CPU check should compare the full read40 arrays against an independently integrated transformed native mask for an asymmetric original view, a recovered view, and a cropped/partial-footprint view, across all eight codes; also compare transformed image/valid alignment. Include origin/border cells. No need to rerun training or add another model.

BN safety is established statically, not by a saved running-buffer check. A small future integration check can assert all BN training flags false and running buffers unchanged around fine_live extraction; the existing exact M3-off replay already tests downstream functional isolation at initialization. M3-off equality across differently trained arms is not expected and must not be claimed.

## Fairness and Same-Weight Diagnostics

The continue_s17 baseline uses the same routed_s17 checkpoint, effective parameter groups, source order/inner IDs, optimizer recipe, 120 updates, source BA/OA/-NLL selection and step0 eligibility. The extra supplement entry in its rate dictionary has no corresponding active parameter group for continue; it is not a fairness defect. Short fine runs change descriptors immediately rather than through zero-initialized residuals, so step0 predictions and source selection can differ legitimately.

The ten-evaluation layout distinguishes: candidate versus continue (training plus reader); candidate versus its own legacyread (fixed weights, both q/ref readers switched); candidate legacyread versus continue (different learned weights under the old reader). The latter two are complementary interventions, not causal mediation proof. A post-training reader swap can itself induce distribution mismatch. Fine-live versus fine-anchor checkpoints are separately selected and are not a same-weight M1 ablation. The new evaluation loop rebuilds reference descriptors after switching readout_arm, which is required and correct.

## Historical Overlap

The 2026-10-31 resolution_matching experiment already implemented this same fine_fixed pooling helper and fine_adaptive native masks. FINDINGS_ZH.md:13-21 describes a frozen older full backbone with an additional 122880-parameter local writer after old M3, 812/318 source organization, 136 updates per fit. It did not replace the existing M3 evidence reader or jointly fine-tune the current routed model.

FINDINGS_ZH.md:44 reports seed17 full OA/BA 47.500/37.435 for coarse soft/pool, fine-fixed soft the same, fine-fixed pool 48.125/37.639, all below the old baseline. Lines54-55 show local-head OA/BA improving from coarse soft 30.625/30.246 to fine-fixed soft 36.250/33.448, with similar improvement for pool. Line72 reports all 160 full predictions unchanged under matched-marginal correspondence removal in five soft models. Thus fine-grid extraction is not new here, previous negative full-model evidence remains, and the present distinct test is evidence replacement plus joint continuation on the current routed model. It does not restore pair correspondence removed by the unchanged aggregate matcher.

## Analysis Delivery

Main-agent completion addendum: verify_geometry.py independently integrates pixel-cell overlaps over the full arrays for original/recovered/partial-footprint conditions and all eight transforms. geometry_checks.json passed: maximum read40 error 2.96e-8; image error below 7.5e-8; footprint error zero. This resolves the geometric-position check requested above without retraining. The reviewer completed final ten-evaluation analysis and inspected gallery02/03; main inspected00/01. See VISUAL_REVIEW_LOCAL.md for actual observations.

analysis.py reads exactly ten completed NPZ evaluations and results.json, validates IDs/labels and cached baseline replay, and reports overall/class/length/class-by-length OA, BA, margins, paired rescue/harm, observation counts and unique ship counts. Length bins are <16, 16-32, 32-64 and >=64, using the maximum actual available-polarization OBB length. There is one seed, not three independent replications.

Four native panels reuse the established CPU draw_panel implementation. Selection prioritizes rescue/harm/stable outcomes, records coverage gaps, and uses the actual legal source16 bank with same-class and baseline-rival examples. All five conditions' scores and all paired outcomes are retained in case_notes_local.json. Case IDs are selected at runtime, never embedded in public code. Gallery and case notes stay local; only aggregate/ is shareable. Visual conclusions remain pending for the later delegated image review.

Commands: `python -B analysis.py --self-test` (in-memory checks only), `python -B analysis.py --check-inputs` (completion only), and `python -B analysis.py` after the main agent's separate ten-evaluation process completes. The reviewer does not wait for training or run final analysis.
