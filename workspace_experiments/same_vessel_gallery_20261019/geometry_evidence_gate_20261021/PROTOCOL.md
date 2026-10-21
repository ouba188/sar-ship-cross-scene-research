# Geometry-guided use of frozen neighborhood evidence

Fixed before fitting on 2026-10-21. This is a bounded classification probe,
not a new convolution, physical compensation, or an independent confirmation.

## Data and rationale

The source-class-transfer study predicts high-response length but sometimes
overcorrects when angle changes without length change. The neighborhood study
shows original R can rescue4002 and harm10461 under a body-only diagnostic,
and adds modest OA to the stronger whole-chip system. Neither result proves
geometry can choose useful classification evidence. Test that missing link.

Use the stronger whole-chip A and frozen original neighborhood residual D from
neighborhood_level_split_20261021. Do not replace them with the weaker body-only
baseline. Native160 crop/pad, no resize, same masks, excluded31237, same4744
test observations/2973ships/18ports8classes, 3identity/datefolds and3paired runs.
This is reused development data, not a strict leave-port/full support-system test.

## Model

Freeze A and D. Current reference is A+r where r=D(x). New score is
A+w*r, w=2*sigmoid(g). Initialize g=0 so the entire old score is retained.
Only a scalar evidence-use gate is learned. No image warp, no predicted length,
no true class as an input, no same-vessel history, no AIS course, no new class
evidence head. A scalar gate can suppress or amplify the existing correction;
it cannot invent a missing class direction.

Content c is the frozen D hidden32 after GELU, standardized using source-train
statistics with sd floor .001, then tanh. Geometry is per-VV/VH cos2phi,
sin2phi,(incidence-35)/10 and valid flags. phi is unsigned OBB axis relative to
available radar axis from the existing snapshot; geometry unavailable means
zero fields, not fabricated image-axis physics. Relative angle is unchanged
by simultaneous coordinate rotation and OBB endpoint reversal. No new metadata
download or claim that geometry is available for all samples.

Five gates: constant scalar; content+validflags; geometry-only; joint content+geometry;
joint with mismatched geometry. Nonconstant gates have nominal40->8 GELU->1
architecture (337 parameters); unused input blocks are zero, but content retains
both geometry-validity flags. Geometry validity also requires usable R. Nominal parameter
counts do not imply equal effective information/capacity. Joint versus mismatched
is the closest same-capacity check for correct geometry correspondence.
Geometry-only is exactly neutral when both geometries are missing. Content and
joint may still use image content under missing geometry. VV/VH conditions are
separate inputs, but the scalar gates their joint residual, not separate experts.

Sham donors permute paired VV/VH geometry jointly within each train/val/test
partition, port/platform/quarter, and geometry-validity pattern. No labels or
outcomes enter permutation. This is a diagnostic control, not a deployed method;
record changed-vector rate, because repeated geometry/singletons reduce power.

## Training and evaluation

Source-only original class-weighted MSE, AdamW .001/.001, batch128,150steps;
same initial seed/minibatches. Source validation selects0/50/100/150 by BA then
Brier. Do not select or tune arms on test performance. 5x3x3=45small fits.
The frozen A/D were trained on source-train too: this probe does not remove
stacking's in-sample supervision issue; inspect source-val versus train behavior.
Keep source-validation zero-step fallback visible, not a learned mechanism claim.

Primary compare against frozen A+r, then content and constant controls. Report
OA/BA/F1, paired MMSI uncertainty, per-class/port, valid-geometry groups, weights
and source-validation behavior. Fixed selected joint with geometry zeroed or
shuffled tests dependence, not unique benefit. Source-mean gate checks whether
sample variation matters. Hold weights/checkpoints fixed for these diagnostics.

Verify zero-init equals old scores, source normalization/selection, no donor
cross-partition or validity changes, frozen-reference replay and selected-head
replay. Inspect actual VV/VH for rescue/harm/CC/WW and old anchors, with exact
gate and class-margin changes. No physical naming from appearance alone.
No further sweeps if geometry fails: locate fitting/use/coverage limits and
retain useful simpler alternatives. Archive positive and negative outcomes.
