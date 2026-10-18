# Axis and radar organized readout

Frozen design 2026-10-19, before new outcome evaluation. This is a targeted
readout experiment, not new backbone training or historical eight-class replay.
Reuse the parent 269 observations, seven fine classes, seven held-date/budget
tasks, five paired nested support draws, queries, source priors and 50/50 fusion.

## Image-grounded hypothesis

Previously reviewed row175 and its actual supports show elongated responses
at different orientations and different surrounding texture. Rows80/236 also
showed that ordinary ROI pooling explains earlier rescues. Thus keep ordinary
target/nontarget readout as the scaffold; do not assume a fixed quadrant is a
common anatomical part. The hypothesis tested is whether centered, unsigned
axis-organized response distributions supply useful additional support relations.

Use the existing frozen detector corners containing the anchor, not AIS heading.
Recover each rectangle's LONG edge from corners. Combine valid VV/VH axes in
doubled-angle space, weighted by confidence and elongation. Record disagreement;
round or conflicting rectangles reduce concentration continuously. The direction
is axial, not bow/stern. Keep the original union target mask to isolate readout.

Four soft angular windows cover opposite directions together, centered at
0,45,90,135 degrees relative to the reference axis. Weights are normalized
exp(2*r*cos(2*(phi-theta-center))), where r is detector-derived axis reliability.
They sum to one at every pixel. Multiply by target and nontarget masks BEFORE
area-downsampling and layer1 pooling, giving eight tokens. Missing axes use
uniform windows and an anchor center; no invented north-facing ship. The image
axis control uses the same center, reliability, masks, token count and normalization.
No pixel rotation or rotation-equivariant backbone is claimed. These are broad
directional response groups, not detected physical ship parts or verified wakes.

Radar directions must be mapped into chip image coordinates from recoverable
product geometry, NOT assumed from UTM image axes. Keep product-record and
pixel-source certainty separate. Do not impute an unknown radar angle as zero.
If only provisional product geometry is available, label that arm exploratory
and report its coverage separately. Incidence angle is an observation condition,
not an in-plane rotation; inspect it separately and do not substitute it for azimuth.

## Fixed insertion and comparisons

For all supplement arms K=.5*Kglobal+.25*Koldgrid+.25*Klocal. Let F be the
previous simple foreground/background kernel, I the image-axis soft readout,
S the detected-ship-axis readout and R the radar-axis readout.

- fg_bg_global: Klocal=F, exact prior-round replay.
- image_soft: Klocal=.5F+.5I.
- ship_soft: Klocal=.5F+.5S; compares to image_soft, not only the old weak grid.
- ship_radar: Klocal=.5F+.25S+.25R.
- ship_image: matched dual-readout replacement using image axis in place of radar.
  Match radar availability in these two arms; no hidden metadata-coverage advantage.
- global_only, global_grid and target_only retain previous comparators.

Radar unknowns fall back to the ship descriptor after normalization. The matched
image replacement has the identical fallback, so paired radar effects cannot
be explained just by that availability policy. Refit SS and QS support paths
together using the unchanged ridge solver. No target-selected weight sweeps.

## Evidence and completion

Verify 180-degree axis invariance, co-rotation of synthetic coordinates and
axes, partition mass/reconstruction, VV/VH corner handling and original feature/
probability replay. Tests target coordinate mistakes and invalid masked pooling,
not an expectation that the trained CNN is rotation invariant.
Report OA/BA, Brier/NLL/MAP, per-date/class, paired support-draw spread, rescues/
harms, missing-axis and geometry coverage. Decompose fixed-competitor score
changes into changed relations and support-coefficient refitting. Review actual
original VV/VH and actual supports for rescue/harm/persistent cases; distinguish
visible morphology, exact score paths and unverified physical explanations.
Archive code, bounded case panels, tables and findings to the research repository.

## Posthoc targeted correction, before its evaluation

The first frozen run showed almost identical radar/image replacement kernels.
The measured April radar image axis spans only -9.265 to -8.816 degrees, whereas
ship-minus-radar angles span 0.657 to 85.802 degrees. Merely adding a radar-frame
kernel does not explicitly condition appearance matching on this relative angle.
This is a reason to test ONE condition-aware interaction, not increase its weight
until a target win appears. Keep all initial outputs unchanged.

For valid geometry define delta=ship_axis-radar_range_axis (mod pi), and
G_ij=exp[.5*(cos(2*(delta_i-delta_j))-1)-.5*((inc_i-inc_j)/10deg)^2].
This is a Gaussian kernel on a circular embedding plus incidence. No AIS,
target label or class prior enters G. Known-known uses G, missing-missing uses1,
known-missing uses0 for the added component only. This preserves PSD and leaves
the shared appearance kernel and whole-region scaffold active across all rows.
Missingness does NOT imply a physical observation condition. Therefore include
an exact availability-block comparator (within-block1, cross-block0).

Four additional fixed arms use Klocal=.5F+.25S+.25H:
- availability_control: H=S times the availability block.
- conditioned_ship: H=S times G (actual appearance-condition interaction).
- relative_angle: same but incidence factor removed, isolate its contribution.
- geometry_only: H=G without S, tests whether geometry-only association suffices.

G is never appended as a class label. Interpret comparisons to both ship_soft
and availability_control. The missingness route is explicitly part of this
diagnostic, not an unqualified fallback guarantee. These are posthoc exploratory
arms, not independent confirmation; no target-best setting is selected. Ten
degrees is fixed as a broad incidence similarity scale, not a learned optimum.
Only samples with valid radar and nonzero detector-axis reliability use geometry.
