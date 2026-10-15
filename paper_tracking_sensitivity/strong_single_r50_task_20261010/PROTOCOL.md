# Task-Aligned Single R50 Pilot

Status: code and CPU validation only. No GPU training was launched by this worker.

## Scope

This bounded OSS2 pilot asks whether source support/query task training helps a
single R50 when training and deployment use exactly the same support operator.
It is not the full strong-single qualification and cannot establish dual-backbone
necessity. Existing full dual scores remain contextual references, not a matched
architecture-only causal comparison against this compact readout.

## Model And Operator

- One ImageNetV2 R50, paired VV/VH at 128x128, existing pixel cache. RGB first
  convolution becomes two channels with the existing mean-and-1.5 conversion.
- Frozen conv1 through layer3, trainable layer4. BatchNorm statistics stay frozen.
  No second frozen R50, SSL encoder, teacher, separate inference feature path,
  learned projection, PCA, frequency module or external metadata.
- Four image rotations run through the same R50. Average GAP2048, then L2
  normalize, multiply by sqrt(32), and append a constant coordinate. Four actual
  image forwards are required; one backbone does not mean one forward.
- A Linear2049x8 source prior uses the same single-backbone geometry. Initialize
  with class-balanced ridge alpha1 and an unpenalized intercept, fitted only on
  permitted source rows. Initialization uses existing frozen C4-R50 pooled2048
  from the same ImageNetV2 paired-input encoder. Before validation/optimization,
  compare actual current C4 geometry with the cache on the first16 inner-training
  rows and save `CACHE_ALIGNMENT.json`: geometry relativeL2/max difference,
  source-prior logit RMS difference and argmax agreement. Historical rounding
  and the current direct-GAP path may differ; the size and cause are not assumed
  in advance. This uses source inputs only, with no second inference encoder.
- Class-balanced residual ridge with lambda8 adapts this source prior using the
  64 support labels. Reuse `aligned_residual_ops.residual_scores` with the actual
  support and query priors.
  Training differentiates through the solve and both support/query embeddings.
  Main query cross-entropy plus fixed0.25 source-prior query cross-entropy both
  use a fixed score multiplier4 and present-class balancing. Optimize layer4
  AND the prior head; the auxiliary term retains direct source-class evidence.
  No multiplier is needed for prediction argmax.
- Frozen comparator uses exactly the same initial R50, pixel recipe, geometry,
  supports and readout, with a prior fitted on all allowed source rows. Adapted
  comparator updates layer4 and that same-sized prior; both have one backbone.

## Source Selection

- Use the exact existing OSS2 source rows and exclusions from `target.npz` and
  `opensarship2_transfer_20261007/source_split.npz`; fail on a mismatch.
- Fixed split seed20261010: choose four entire source ports with at least128
  allowed rows, without labels. Purge their known MMSIs and product identifiers
  from inner training. Missing identities are not treated as a shared vessel.
- Validation has two fixed episodes per held port,64 support and up to128 query
  rows. No PCA/scaler is fitted. The source prior is fitted ONLY on inner training
  rows, never on held-port labels. Within an episode supports and queries are disjoint rows; distinct
  vessel identities within that episode are NOT imposed, matching the existing
  target support protocol rather than claiming unseen-vessel evaluation.
- Source training uniformly samples eligible ports, then64 support+32 query rows.
  Default6 epochs x32 episodes, AdamW lr1e-4/wd1e-4, cosine schedule and gradient
  norm clipping5. This is a bounded pilot, not established convergence.
- Select highest mean validation port/draw BA; ties keep earliest epoch, including
  epoch0. Save the complete curve. Best-at-budget-boundary is explicitly flagged.
- Restart the ImageNet tail, refit the initial balanced prior on all allowed
  source rows, and task-train both for the selected epoch count, retaining the
  same six-epoch LR schedule prefix. Refit does not use
  target images or labels. Save both trained tail and trained prior, plus the
  initial all-source prior for the frozen comparator. Epoch0 selection gives a
  genuine unchanged comparator.
- Default one neural seed20261010. `--seeds 20261010 20261011 20261012` enables
  separate source selection/refit for three neural seeds; the five target support
  draws are not five neural seeds.

## Target Evaluation

Load only the established metadata, source rows,5x64 support labels and query
masks. Persist target predictions before loading target query truth. Report BA,
OA, per-class recall and classwise rescues/harms against the frozen matched arm.
Both arms extract current actual pixels; no old cache feature precision mismatch.
Saved extraction timing is throughput bookkeeping, not an isolated paired
end-to-end latency benchmark. No independent-confirmation claim is made.

## Commands

Interpreter: `E:/Hermes/paper_tracking/envs/msfa-openmmlab-py310-rtx50-cu132/Scripts/python.exe`

CPU-only commands (safe while another task uses GPU):

```powershell
& $python E:/Hermes/paper_tracking/sensitivity/strong_single_r50_task_20261010/test_strong_single_r50_task.py
& $python E:/Hermes/paper_tracking/sensitivity/oof_reliability_check/strong_single_r50_task.py prepare
```

After the main worker releases the GPU:

```powershell
& $python -u E:/Hermes/paper_tracking/sensitivity/oof_reliability_check/strong_single_r50_task.py all
```

Estimated one-seed wall time: roughly10-30 minutes on the RTX5050 laptop, depending
on memory pressure and selected refit epoch. This is an estimate, not a measured
GPU runtime. The first8-episode log provides an actual timing estimate. No large
prefix cache is produced; checkpointing bounds training activation memory.
The pilot stores two layer4 checkpoints and two target feature arrays per seed,
roughly200MB/seed. It does not copy whole source pixels or full pretrained weights.

## Remaining Boundaries

This adds meaningful task-aligned tail adaptation, not full-network optimization,
SAR pretraining parity, matched full-model module coverage, three completed neural
seeds or all25 tasks. A negative result must first inspect finite gradients,
actual weight changes, validation trajectory and selection/refit behavior. The
fixed kernel scale and lambda are not target-tuned; this single recipe does not
exhaust strong-single designs. Do not use it alone to defend the dual backbone.

## Superseded Design, Not Run

The original zero-prior design was NOT GPU-trained or evaluated. Known OSS2
support-label draws omit class6 in draw1 and class7 in draws2/4 (zero-based).
With no source prior, unseen classes have identical desired support columns and
cannot retain distinct source-class evidence. This concrete closed-eight-class
failure prompted the above prior restoration before target prediction. It is
an unrun superseded design, NOT a negative experimental result or target-query
selection. The support labels are the already authorized64 labels.
