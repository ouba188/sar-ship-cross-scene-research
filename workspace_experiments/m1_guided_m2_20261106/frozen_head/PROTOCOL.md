# Frozen-head continuation: M1-guided M2

2026-11-06. Authorized local continuation; no formal replacement, no Git sync.

## One changed variable

Freeze the original classification head and train only the additional regional
readout. Reuse the previous study's float32 frozen cache, source719, development228,
historically exposed evaluation378, fixed32 source references and upstream seed17.
All other operator definitions, initializations, losses, learning rates, batches,
steps and seeds remain unchanged. The source-only difficulty weights also remain
anchored to the same frozen model. No checkpoint selection using evaluation.

Three arms (mean/content/class) x seeds17/29/43, fixed180 steps, batch24, 4320
presentations each. Head LR is removed rather than changing readout LR. Readout
value/scorer LR3e-4, output LR1e-3, AdamW decay.01, clipping5. Initialization of all
parameters occurs before head freezing, preserving previous random draws.

## Why this experiment

The preceding round's class readout improved evaluation BA relative to its own
updated-head path with the readout disabled, but the updated head itself lost BA
relative to the original model. Freezing the head tests whether preserving the
original decision rule makes the new evidence useful overall. It does not assume
that removing one loss component guarantees better jointly learned updates.

## Comparisons and checks

Report paired frozen-head versus updated-head outcomes for each arm and seed,
class versus trained mean/content controls, and every arm versus the original
model. Report OA/BA/NLL, per-class changes and unique rescued/harmed observations.
Off must reproduce original predictions; head parameters must change by exactly
zero, with no head gradients and nonzero readout gradients. Verify identical old
and new training orders. Preserve same-weight uniform/shuffle diagnostics, but
do not mistake them for independently trained controls.

Inspect native VV/VH queries and legal source reference images for direct
rescues/harms and persistent correct/wrong examples. Exact linear head regional
contributions identify the model-level score path, not a physical scattering law.

Scope remains shallow layer2 regional output supplementation, not an internal
M2 writeback redesign. No M3/frequency/backbone changes. Only the added readout
seeds vary; this is not three independent upstream backbone training runs.

Post-run note: the unchanged global gradient clipping implementation acts on
trainable parameters, so freezing the head removes its gradients from that norm.
The saved first-two-step gradients confirm changed readout gradient scaling in
some paired runs. This estimates the implemented freezing policy, not an isolated
decision-boundary effect. See FINDINGS_ZH.md; no silent retraining or correction.
The user subsequently authorized synchronizing the M1/M2 code and these records.
