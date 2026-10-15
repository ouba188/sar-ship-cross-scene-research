# Frozen source-to-FUSAR transfer, 2026-10-07

## Fixed question and policies

Test whether the current same-64 adapted base plus at most 5% expert handoff
adds value when transferred to a compatible subset of FUSARShip v1.0.
The three policies are original_base, base_visual64, visual_handoff5.
No new module, parameter search, or outcome-based cohort changes.

Reuse PCA128/PCA64, balanced source ridge alpha=1, source-port OOF four-feature
logistic gate C=1, both residual ridges strength=8, positive gate threshold=0,
observed-expert-class guard and floor(0.05*N_eval) admission cap. Both branches
use the SAME 64 target annotations. Five uniform no-replacement draws use
seeds 0..4; no class quotas or rare-class resampling. Save every policy before
query readout and preserve all five results.

## Target cohort and identities

Use experiment_cohort.csv from bdh_fusar_preparation_20261003. Honor its
eligible=True boundary: 3166 of3276 original five-category chips, after96
shared-main-vessel and14invalid-identity exclusions. No additional quality,
size, AIS-dimension, or interference filtering. Fine subcategories must be
explicitly compatible with source class IDs; generic CargoShip/Tanker and
unsupported/ambiguous types remain excluded. Reject contradictory official
coarse/fine categories rather than relabeling them from model outputs.

Keep all eight source output logits even if fewer classes survive target
alignment. Do not mask predictions to the query's known ground-truth class
set. Balanced accuracy averages only classes present in the evaluated query,
and absent class recalls are null, not zero. Report complete mapping coverage
and class counts; this is NOT the original full five-category benchmark.

Check source MMSI overlap against ALL eligible target rows before fine-label
selection; remove any remaining overlaps before refitting source PCA/heads
and the 24-port OOF gate. Existing target eligibility already excluded shared
main identities. FUSAR scene IDs are unavailable: filename class/subclass
codes must not be repurposed as scene or port identifiers. Source inner OOF
continues to purge both source MMSIs and source scene IDs as before.

Target support/query remains row-level, matching the current main method.
It is not a vessel-disjoint target test or a within-FUSAR cross-port test.

## Single-image input adapter

Current source base: true paired VV/VH, two-channel ImageNet R50 V2, 2048D.
Current extra representation: source VH via domain-SSL R18 e206, 512D.
FUSAR provides one image here. Copy the same image into both base input
channels and use the same image for SSL. This zero-parameter input adapter
does NOT recover a missing polarization, create new information, or validate
dual-polarization complementarity. Call it single-image dual-encoder transfer.

Reuse the current encoder implementations and preprocessing. The existing
feats_fusar.npz has a different base/normalization and is not a substitute.
Extract this cohort once, no backbone training. The same extractor and
checkpoint were source-replay checked in the preceding OSS2 experiment;
no repeated encoder audit is required absent a changed implementation.

## Readout and claims

Report BA/accuracy, all per-class recalls, admission counts and the paired
full-minus-adapted-base increment for all five draws. Five overlapping draws
are not five independent datasets. Main result is a positive, null, or negative
fixed transfer outcome, not a guarantee that this dataset will show gains.

FUSAR has already been used in development, including external_os2_fusar.
This is not a new blind confirmation set. The SSL encoder retains its F0921
pretraining exposure. Budget is expert admissions, not encoder inference cost.
Original OSS2 and all earlier outputs are preserved unchanged.
