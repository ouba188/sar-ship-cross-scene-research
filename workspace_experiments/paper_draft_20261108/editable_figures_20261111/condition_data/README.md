# Condition effect data for editable figures

## Recommended figure

Use `condition_long.csv` (144 cells), or `condition_matrix.csv` (24 rows x 6 class columns).
Two facets: `joint` and `single`. Each facet contains all 12 combinations of VV/VH,
body/side/axis_end, and fine/mid. All six observed classes, negative effects, near-zero
effects, and positive effects are retained. The single facet means single VV for VV
rows and single VH for VH rows; it is not one fixed single-polarization head across rows.
Both single and joint heads retain four geometry features. Historical diagnostic,
NOT the current Full model. Both facets may share a zero-centered color scale.

Primary cohort: round26 `new_coverage`, high preset dose=1.0, 1165 observations,
969 vessels, 22 overpasses. This excludes six old observations, not all old vessels:
five retained observations belong to four old-case vessels; only 965 identities are new.
No selection uses effect sign or predicted success. Main rows use vessel-equal means:
first average all observations of each MMSI in the cell, then average those vessels.
Repeated doses, conditions and heads are never counted as independent samples.

`condition_n_ships_matrix.csv` and `condition_n_observations_matrix.csv` align exactly
with the effect matrix. The long table also gives overpass/fold counts, actual readout,
model, dataset, source CSV, metric definition, L2 range, and rescue/harm event counts.
Show small samples explicitly: crude-oil tanker has 11 vessels and fishing has 4;
`small_n_under_20_ships` is a descriptive plotting flag, not a statistical cutoff.
Missing classes tug/offshore supply have no eligible observations, not zero effect.

## One margin definition

For each observation let c0 be the strongest incorrect class under the ORIGINAL JOINT
head. For either readout h:
  delta_m_h = [z_h,y(I(x)) - z_h,c0(I(x))] - [z_h,y(x) - z_h,c0(x)].
Round26 uses saved `delta`; round22 uses saved `delta_fixed`. Positive values mean
this attenuation helps that fixed true-versus-rival score difference. They are raw
ridge decision-score margins, not probabilities or OA/BA percentage points.
`own_delta` and `decision_margin_delta` are deliberately not used in the matrices.
Readout changes include the already-fitted alternative diagnostic head, not just
removing a branch at inference; this preparation never fits or loads any model.

The saved same-mean control appears separately as `level_delta_margin` and
`residual_delta_margin = delta_margin - level_delta_margin` in long tables. The main
matrix uses raw delta_m only. A residual is not pure frequency causality: the control
matches mean change but not L2 or the full pixel distribution.

## Operation, dose and geometry

fine = I - Gaussian_sigma1(I); mid = Gaussian_sigma1(I) - Gaussian_sigma2(I),
sigma1=1 and sigma2=2 native pixels, reflect boundary. These are overlapping local
spatial-scale components, not independent electromagnetic frequency bands.
For mask M: taper=min(distance_transform(M)/2,1)*M;
d=clip(I-0.5*taper*component,0,1)-I; I'=I+dose*T*d/||d||2.
Round26 T=0.5*minimum proposal L2 over BOTH polarizations, all three regions, both
scales, and own/common masks. Only common-mask operations are evaluated. Dose 1.0
therefore means the higher preset matched-L2 dose, NOT complete removal or identical
absolute L2 across observations. All 12 conditions have equal L2 within an observation.
Round22 uses its own polarization mask and matches L2 across six region/scale
conditions within each polarization separately. Its matrix must remain separate.

body: detection OBB mask. axis_end maps raw `end`: outer strips extending 8 native
pixels beyond the axis ends, with transverse extent width/2+2, excluding body.
side: lengthwise span |u|<=length/2 and width/2<|v|<=width/2+8, excluding body.
Round26 uses intersection of corresponding VV/VH masks (>=8 pixels per region).
axis_end is not an identified bow/stern or wake. mid maps raw `middle`.
Original native crops/padding and geometry are retained; no new image processing occurs.

## Other deliverables

- `condition_all_doses_scopes_long.csv`: all 576 round26 aggregate cells, both doses
  (0.5/1.0), all-eligible and new-coverage scopes, both readouts. Do not pool scopes.
- `round26_selected_observations.csv`: the 27960 main-scope effect records, with MMSI,
  observation/fold/rival, L2 and original CSV line for traceability. Local data only.
- `condition_readout_difference_long.csv`: 72 paired ship-mean joint-minus-single
  effects. This is an alternative contrast, not the raw delta_m facet.
- `round22_condition_long.csv`: all 336 cells, 7 classes, two doses and scopes;
  `round22_condition_matrix.csv` and matching count matrices show only dose=1,
  expansion_only (209 observations, 104 ships, 31 overpasses, six ports).
  Historical full_global diagnostic, not the current model or the entire 24-port set.
- `validation.json`: score-based definition checks, paired-readout checks and exact
  comparisons against original class_detail/by_class/summary CSVs. No new CI is made.
- `provenance.json`: source files, code locations, definitions, exclusions and output map.

## Reproduce

Run `python prepare_conditions.py` from this figure directory, or use its absolute path.
Only Python's standard library is needed. The script reads existing CSV/CSV.GZ files
and writes only this condition_data directory. It imports no original experiment
modules, reads no weights/features/images, and performs no inference or training.
Validation fails before export if definitions, counts or report agreement disagree.
The original per-class high-dose estimates reproduce within 1e-12. Class composition
and small cells remain visible; no significance or natural physical cause is inferred.
Existing case reviews are the original rounds' CASE_REVIEW_ZH.md; this data-only task
does not claim a new raw-chip inspection or a newly run experiment.
