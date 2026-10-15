# SARCLIP frozen-backbone screening, 2026-10-10

Status at definition: real CUDA smoke passed; no new target predictions inspected.
Input correction before any new SARCLIP target predictions: native Byte-pair
layout follows the official `sar_clip/data.py::readtif`, not two independent
gray RGB views. See INPUT_CORRECTION.md; common-readout/R50 protocol unchanged.

## Question and scope

Can a locally available SAR-pretrained visual backbone compete with the current
dual system on the same paired-pixel cohorts and five sets of 64 target labels?
This is external-pretraining competitiveness, not matched-pretraining architecture
attribution or a claim that the single backbone has been fully task-optimized.
Initial scope: 23 eligible 24-port tasks plus OpenSARShip 2.0. F_0921 follows later.

## Frozen extraction

- BiliSakura SARCLIP ViT-L/14 local safetensors; official repository visual module.
- Visual only, 303,966,208 parameters; all 296 visual state tensors loaded strictly.
- The same existing 128x128 uint8 VV/VH pixels used by the current experiments.
  Form the official three-channel Byte-pair layout(VV,VH,mean(VV,VH)), bilinearly
  resize to224, and map from0..1 to-1..1. No additional native-resolution information.
- Both polarizations enter one shared visual backbone in one forward per chip.
- Projected768-vector, L2 normalized. No target-selected polarization.
- bfloat16 inference; no C4 and no neural optimization in this screening.
- No text tower, prompts, target query labels or target transductive fitting.
- Cache on D:; retain historical features and predictions unchanged.

## Source-selected common readout

For each outer task use exactly its stored permitted source rows. Compare the
SARCLIP representation and existing C4-R50 pooled representation under the same
new readout and the same nine configurations. The latter is a matched-readout
control, not a newly optimized strong R50 neural backbone.

Use row-normalized features; subtract source mean and divide by a single source
RMS vector norm. Add a constant coordinate only for the support residual kernel.
Fit balanced source RidgeClassifier on inner-source rows. Source ridge alpha
in(1,10,100), residual lambda in(0.1,1,8). Equal tuning opportunities for both
representations. No target-label checkpoint/parameter selection.

Choose three validation ports at evenly spaced indices in the alphabetical list
of permitted source ports with more than128 samples. For each, remove that port
and any same-MMSI observations from inner training. Fit all source statistics
and source heads on those inner training rows only. Use three deterministic
random64-label support draws within each validation port; score other samples
using the existing balanced residual-ridge operator. Select highest mean BA,
ties follow the stated alpha then lambda order. These are support-query tasks,
not a claim of vessel-disjoint support/query acquisition validation.

Refit the selected source geometry/prior on all permitted outer-source rows.
Use precisely the stored target support, support_labels and eval_masks; every
module shares those64 labels. Freeze predictions before reading target query
labels. Report OA, BA, per-class recall and rescue/harm against the matched
source-refit current dual reference. The dual is a reference with additional
modules/training, not an equal-compute or pretraining-matched control here.

## Boundaries

All cohorts have been used for development. SARCLIP pretraining overlap with
these cohorts is not ruled out. Five support draws are not five neural training
seeds. No physical specialization, causal mechanism or irreducibility claim.
Warm extraction throughput is not full deployment latency. A weak screening
result cannot prove that SARCLIP or single backbones are intrinsically inferior.
