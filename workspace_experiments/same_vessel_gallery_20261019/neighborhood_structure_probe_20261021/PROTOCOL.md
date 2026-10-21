# Neighborhood amplitude and arrangement probe

Fixed before results on 2026-10-21. This is a mechanism probe, not a proposed
deployable shuffle module or a physical SAR scene intervention.

## User correction before training results

The user identified the unchanged whole-chip baseline as an alternative route
for original neighborhood information. Retain that experiment only as a system
incremental-utility test. ADD a separate body-only mechanism track, not overwrite
or reinterpret the whole-chip experiment. This amendment was made after feature
extraction and before any trained outcome was inspected.

Use the already source-trained `crude_input_controls/body_global` classifier:
its input zeros non-target pixels BEFORE convolution and was fitted with that
same input. No real non-target pixel values enter this baseline. It still uses
OBB-derived dimensions and valid-footprint pooling, so this is not a mask-free
or metadata-free classifier. No test-time-only masking of a whole-chip model.

Freeze that classifier, separately train original centered-R+mean and all four
transforms with exactly the same head/budget/source selection (45 fits). In
this track, neighborhood pixel values enter only the independent R channel.
Apply fixed-head tests using this track's newly trained original-R head, never
reuse a residual fitted against a different baseline. Compare arms within this
track; do not equate its lower body-only baseline with the stronger whole-chip
model or attribute between-track absolute gaps solely to background removal.
Separate results live in `body_only/`. This closes the original-pixel bypass
for the mechanism question without claiming physical causal identification.

## Prior evidence and question

The previous D residual improved OA by .548 pp on the stronger native frozen
classifier, but its explicit neighborhood mean had little net utility. Cases
4002 and 10461 show different extended bright responses; local removals can
help or hurt. We ask whether useful supplementary information depends on local
amplitude, fine arrangement, or organization beyond a local neighborhood.

In the WHOLE-CHIP track, the original classifier sees the unchanged chip in EVERY arm. All changes
below apply only to the added R channel. Therefore conclusions concern extra
information for this frozen system, not the necessity of these features for
all SAR classification.

## Data, reference, and interventions

Reuse exactly the old cohort, excluded observation31237, masks, 160px native
crop/pad, missing/conflicting view handling, source-only splits and3 seeds.
R excludes accepted target and other OBBs and padding; not necessarily water.
Whole-chip reference is previous `neighborhood_level_split_20261021` D separated;
the body-only track trains its own original-R reference as amended above.

Use original PNG x, R mean mu, and encoded z=.5+.5*(x-mu)*R.
1. half_amplitude: z=.5+.25*(x-mu)*R. Mean and ordering are unchanged; local
   deviations halve, variance quarters. No clipping or per-image std division.
2. tile_shuffle: permute ONLY R pixels independently within each fixed native
   16x16 tile. Preserve each tile histogram, coarse tile means, R mask and
   overall histogram. Destroy fine arrangement inside tiles, not all texture.
3. global_shuffle: permute ONLY R pixels across the full R region. Preserve
   total histogram/mean/variance and masks, but destroy original spatial layout.
4. flat: encoded image everywhere .5. Retain original mean slots, availability
   and the pooling mask. Tests how much mean/mask/constant corrections can do
   without neighborhood image variation, not a pure shape-free baseline.

Shuffle RNG depends only on observation id, polarization and run seed; it
never accesses class/port/partition/outcome. Three realizations correspond to
the three paired training runs. Variance across runs includes old/new training
initialization AND shuffle realization, not isolated optimizer variance.
No pixel resampling, resizing, physical unit assumptions or target deletion.
The artificial permutations create new edges: matched retraining reduces the
train/test mismatch but does not make the intervention physically realistic.

## Two evaluations

Fixed-head sensitivity: keep D source normalization and D parameters, replace
only its supplemental descriptor, leave original mu slots and original strong
classifier scores unchanged. No new fitting.

Matched training: each variant receives its own source-training feature
normalization, identical260->32->8 head,8616 nominal parameters, optimizer,
sampling,150steps,weighted MSE, and source-validation selection0/50/100/150
by BA then Brier. Same initial seeds and minibatches as original D; 4x3x3=36
new fits. No test selection. Flat has the same nominal architecture but less
effective information, which is intentional.

Reproduce the source-training/validation rows and frozen A/D scores. Reuse D
without retraining because its settings and raw input remain identical.
Do NOT change old run.py globals or overwrite existing output folders.

## Interpretation and checks

- Fixed-head degradation alone means dependence, not irrecoverable information.
- Matched degradation strengthens an information/representation-loss hypothesis
  within this fixed stem, head, budget and validation procedure; it does not
  identify physical causes or prove a universally necessary feature.
- Half-amplitude is an invertible input rescaling, NOT information deletion.
  Its effect concerns the fixed nonlinear stem and downstream learning. Tile
  versus global also changes local distribution heterogeneity; it cannot alone
  identify long-range spatial relations. The current pool retains no tile IDs.
- Tile vs global distinguishes remaining coarse organization under these
  operators, but pooling, artificial edges and optimization remain relevant.
- Do not infer identical information from a nonsignificant difference.
- Check actual histogram preservation per usable view, tile histograms, no
  non-R changes, mean/variance scaling, original descriptor replay, native
  pixel preservation, fixed score replay, source-only normalization/selection.
- Report pooled OA/BA/F1, individual seeds/folds/classes and paired ship CIs;
  inspect original VV/VH and actual transformed inputs for previous anchors
  plus current rescues/harms/persistent outcomes. No physical naming from looks.
- This is reused development data, not an independent acquisition test. Near/
  far matched-area comparisons are not part of this bounded round.
