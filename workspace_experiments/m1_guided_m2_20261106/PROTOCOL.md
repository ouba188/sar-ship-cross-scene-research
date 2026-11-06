# Frozen M1 responses guiding shallow M2 regional pooling

Date: 2026-11-06. Local pilot; no formal replacement and no Git sync.

## Hypothesis and scope

Within a geometric region, M1 candidate-class response patterns may help retain
discriminative feature differences that regional averaging suppresses. Native
source example000586 has nonuniform localized VV/VH responses, but appearance
alone does not establish a class-specific or physical mechanism. Earlier M1
rescues often survived covariance removal; this experiment therefore tests an
explicit M1-to-M2 information interface, not another claim about covariance.

Freeze the existing source719 small_class_s17/final.pt from m1_coverage_20261101.
Modify only shallow layer2-output regional readout (512 channels,20x20). M1 is
inside layer2[3] after conv2; its response grid and downstream conv3 output share
stride/origin. Keep deeper M2, internal M2 writeback, backbone, and protected M3
reference message fixed. The public carrier does not include private layer2[2]
M2. No new frequency branch, geometry metadata, labels, or references.

## Three paired arms

- mean: repeated regional mean descriptors.
- content: four class-agnostic attention maps learned from local normalized
  layer2 features.
- class: four frozen M1 candidate-class relative response maps. These are NOT
  certified pixel labels. For each pixel subtract the other-class mean; normalize
  by valid-image spatial standard deviation with floor, then bound with tanh.

All arms share the same structural mean path, feature512->32 value projection,
five32D tokens (mean plus four weighted summaries), region output160->512, and
bounded residual gain0.1. Outputs initialize to zero. Separate VV/VH parameters.
Softmax is restricted and normalized within each soft geometric region; empty
regions yield zero weights/update. Keep regional identity (body/end/side) by
injecting each output into its original region feature slot. No query labels are
fed into attention. Content has an extra trainable512->4 scorer; report this
capacity difference rather than claiming identical effective capacity.

## Dataset and optimization

Use source719, historical development228 and fixed evaluation378, fixed32 source
references from the previous study. Added selection194 is NOT used for this fit.
The frozen carrier was trained on719, not913. These evaluation data have historical
research exposure and are not a fresh blind test. Maintain ID/MMSI/orbit isolation.

3 arms x3 new-training seeds17/29/43 =9 fits,180 optimizer steps, batch24, cycling
through shuffled permutations with full-size batches across permutation edges.
Same batch order, source class/vessel weights and frozen-carrier difficulty weights
for all arms. Train new pooling residual and final head only. Value/scorer LR3e-4,
output LR1e-3, head LR1e-5, AdamW decay0.01, gradient clip5, weighted CE. No new
auxiliary loss. Fixed final step180; intermediate dev reports do not select weights.
The three seeds vary this new readout/head training, not the frozen upstream M1.

## Required evidence

Compare class vs mean AND class vs content, OA/BA/NLL, per-class rescues/harms.
Compare trained heads with their own residual disabled to separate head fitting
from the added path. For class, use same-weight uniform attention and a within-
region permutation preserving weight histograms to test correspondence reliance.
These interventions are diagnostics, not substitutes for the trained controls.
View native VV/VH queries and actual legal reference chips for rescued, harmed,
stable-correct and stable-wrong cases; inspect attention/region overlays for selected
cases without interpreting attention as proof of causality.

## Specific implementation checks

Compare hooked live feature/head input against the frozen full forward and saved
previous predictions. At initialization all three arms must reproduce frozen
scores. Verify invalid/empty-region weights, independent weighted-sum equivalence,
head slot update order and nonzero new-path gradients. Verify checkpoint replay.
Caching is exact float32 evaluation-mode algebra of this frozen readout experiment,
not a substitute for training a changing backbone. No further sweeps this round.
