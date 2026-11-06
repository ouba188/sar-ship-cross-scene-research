# M1 coverage and trained covariance replacement

This is a new local experiment, not a formal-model replacement. No Git sync.

## Question

The earlier class-competitive M1 package beat tuned mean on historical OSS2
development228, but setting covariance to zero at inference changed no top1
predictions. Train-time covariance removal and expanded independent source
coverage are therefore tested together. Native VV/VH examples show overlapping
bright structures across classes; they motivate richer relations, not a proven
physical explanation or guaranteed improvement.

## Data change

Compare source719 against source719 plus historical selection194 (913 total).
Selection194 is explicitly reassigned to training and is never used to select
the new models. Keep development228 and evaluation378 untouched. Fixed source
references32 are identical. Check sample, vessel and platform-orbit separation.
The extra194 contain193 new vessels and5 overpasses, of which4 are new; the
source pool increases from13 to17 platform-orbits. One shared orbit has a
different datatake, which is not counted as another independent overpass.
Evaluation378 has historical research exposure, so results are frozen coverage
confirmation, NOT a new blind test. Preserve all earlier role files and results.

## Matched training

Mean, class-competitive M1, and class M1 with covariance input zeroed from the
first training step. The latter preserves response maps, local gating, trainable
biases, mean path, auxiliary objective and tensor dimensions. It removes access
to covariance in BOTH feedback and auxiliary prediction. Its covariance-only
weights necessarily receive zero gradients; do not claim equal effective capacity.

Use previous early position layer2[3] post-conv2/pre-conv3, one M1, separate
polarization parameters. Gains fixed from the earlier pilot: mean0.1, class0.3,
no_cov0.3. Train M1 + adjacent conv3 + head, M2 and protected M3 message frozen;
public carrier still lacks private layer2[2] M2. No new frequency branch.

Each fit180 steps, batch24, permutation cycling; same per-cohort/per-seed batches
for all arms. Seeds17/29/43, 18 fits. Report exact consumed examples because a
permutation's last batch can be shorter. Select fixed step180, never a test best.
Internal LR3e-4, output1e-3, conv3/head1e-5, AdamW decay0.01, clip5,
classification CE +0.02 auxiliary CE. Source class/vessel balancing and shared
no-M1 carrier margin<1 hard2x weighting are recomputed per legal source pool.
This slightly differs from the earlier pilot's historical difficulty carrier;
all conclusions use within-round paired controls, not old/new score subtraction.

Primary: class vs mean and class vs no_cov, then their changes with training
coverage, on both fixed query cohorts. Report OA/BA/NLL, per-class/pass counts,
rescues/harms and native VV/VH cases with actual legal reference chips. Do not
select the best dataset or seed afterward. No additional epochs or tuning based
on evaluation378; a follow-up training-duration question would be separate.

## Targeted checks

Native new cache must reproduce previous cached rows and source prefix; this
catches index/preprocessing mismatch. Zero-init paths must agree; trained no-cov
must actually remove covariance and its gradient. Checkpoint reload and identical
batch orders catch replay and pairing errors. No full dataset rewrite or label edit.

## Runtime adjustment

After the first three719/seed17 fits, severe host-memory paging made random
memmap reads slow. The unfinished913 mean run was stopped and preserved under
interrupted_io_expanded_mean_s17; it is excluded from all result tables. Exact
float32 inputs are staged on GPU for remaining runs, with before/after prediction
parity checked. Completed fits are reused. No training setting or precision changed.
