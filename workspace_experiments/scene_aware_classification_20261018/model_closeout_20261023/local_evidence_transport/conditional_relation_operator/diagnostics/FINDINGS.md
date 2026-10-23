# Conditional relation diagnostic: utility644 and frozen bank64

Completed 2026-10-24. Only this diagnostics directory was written. No training, classifier, hyperparameter search, selection/evaluation prediction access, permutation enumeration, Git, or changes to prior modules. Read prior pairing_identifiability and pairing_excess implementation/findings. ClearSAR run completed successfully; measured main execution 21.56 s after imports.

**Directional agreement carries class information beyond norm or singular-value shape. Normalizing C improves all eight group/reference-source macro contrasts on the comparable gap/RMS scale, but removes some class-specific amplitude advantages. Fixed within-image ridge is modestly predictive on held-out edges; the full 32-dimensional conditional map remains underdetermined.** These are source descriptive findings, not classification gains or cross-ship map generalization.

## Data and definitions

- Utility: 644 images, 587 MMSIs, 45 passes. Frozen bank: 64 images, 64 MMSIs, 35 passes. Union 708 images. Only these rows enter projection/statistics; the shared NPZ is the existing storage container.
- Existing layer1 PCA32 projection and unit-normalized tokens are frozen. Legal edges require positive mass at both endpoints; mass is not an extra regression weight. For each image/group, centered B,N give Cbn=B.T@N/m and Cbb=B.T@B/m.
- VV_side/VV_end have 644 valid queries and 8/4 edges. VH_side/VH_end have 561 valid queries; 83 missing are excluded, not zero evidence. VH_side has 8 edges; VH_end has 4 for 560 images and 3 for one image.
- Both reference pools exclude same MMSI OR same pass. Each reference class is averaged independently; query gap is same-class mean minus the mean of the seven other class means. Eligible queries have all eight reference classes available. Final macro is the equal mean over eight query-class gap means. No class argmax is computed.
- Raw kernel is <Cq,Cr>F. Cosine kernel divides by ||Cq||F ||Cr||F. Amplitude-only kernel is that norm product. Singular-values kernel is the dot product of descending singular values, an orientation-free comparison; its cosine version removes overall amplitude. These are comparisons, not additive components or causal attributions. All use identical eligible pairs.
- gap/RMS divides each macro gap by that kernel's RMS across eligible ordered pairs. It makes scales comparable but is not an accuracy, effect probability, or explained-information fraction. Pair RMS is pooled; class gaps are macro-balanced. Original gaps are also retained below and in CSV.

## Actual reference coverage

Every valid query remains eligible. For each VV group, each bank class contributes exactly 8 references (64 total). For each VH group, class IDs 0..7 contribute **8,5,7,7,8,8,7,6** references (56 total). Exclusion causes no further bank reduction in this cohort. Utility reference counts per query/class range 19..204 for VV and 16..198 for VH, with medians 45 and 33. `reference_coverage.csv` records every query/class count, including invalid queries.

Class IDs: 0 bulk_carrier, 1 fishing_vessel, 2 general_cargo, 3 product_chemical_tanker; names for remaining classes are preserved verbatim in image_diagnostics.csv. Counts are observations, not independent-ship sample sizes.

## Raw versus normalized class kernels

| Group | Refs | Queries | Raw macro gap | Cosine macro gap | Raw gap/RMS | Cosine gap/RMS | Norm-only gap/RMS | Spectrum gap/RMS | Unit-spectrum gap/RMS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| VV_side | utility | 644 | .000743 | .030979 | .178835 | .227033 | .022138 | .023410 | .001519 |
| VV_side | bank64 | 644 | .000646 | .029104 | .161242 | .215580 | .014186 | .015554 | .001951 |
| VV_end | utility | 644 | .000421 | .012417 | .123193 | .158642 | .015207 | .017915 | .002523 |
| VV_end | bank64 | 644 | .000201 | .009659 | .063045 | .126968 | .003978 | .006346 | .003224 |
| VH_side | utility | 561 | .000701 | .026851 | .142686 | .168119 | .022364 | .023528 | .001340 |
| VH_side | bank64 | 561 | .000494 | .017498 | .102618 | .113157 | .017912 | .018525 | .000837 |
| VH_end | utility | 561 | .000197 | .006006 | .066252 | .079516 | .019217 | .023017 | .004263 |
| VH_end | bank64 | 561 | .000104 | .003590 | .033481 | .047726 | .012176 | .014289 | .001848 |

The utility raw gap/RMS reproduces the earlier pairing-identifiability values. Norm and spectrum have positive but much smaller relative contrasts; spectrum shape alone is weak here. Bank contrasts are smaller than utility contrasts despite full class coverage, especially for ends. Coverage alone does not guarantee representative directions.

**Do not conclude amplitude is useless:** per-class norm distributions differ. VV_side fishing mean/median are .10146/.08438 versus chemical tanker .17849/.17915. Normalization reduces 28 of the 64 per-class gap/RMS contrasts. In VH_end-to-bank, tug/towing class 6 changes from +.03352 to -.00818; general cargo drops .21528 to .05023. Macro improvement is compatible with loss of useful class-specific amplitude. No classifier comparison establishes that retaining or discarding amplitude improves final decisions.

All norm distributions by class, reference pool, and eligible-query population are in `norm_distributions.csv` (n, mean, p10, median, p90). Eligible-query populations equal valid utility populations here:

| Group | n | Norm mean | p10 | Median | p90 |
|---|---:|---:|---:|---:|---:|
| VV_side | 644 | .15840 | .09701 | .16272 | .21132 |
| VV_end | 644 | .17555 | .09704 | .17118 | .25702 |
| VH_side | 561 | .16634 | .11161 | .16998 | .21207 |
| VH_end | 561 | .17672 | .10628 | .17228 | .25508 |

## Is a conditional map observable with 8/4 edges?

Computed the requested A=(Cbb+lambda I)^-1 Cbn with lambda=trace(Cbb)/m, independently for each image/group, without labels or fitted hyperparameters. A maps a centered body row to a neighbor row. Full-image operator norms, ranks, effective ranks, and ridge degrees of freedom are saved.

For each leave-one-edge-out fold, remove the edge first, recompute both means and covariances using k=m-1 edges, set lambda_fold=trace(Cbb_fold)/k, and predict Nheld=mean_Ntrain+(Bheld-mean_Btrain)Atrain. This applies the same fixed formula at the actual fold sample count. Baselines are the training-neighbor mean and the neighbor token paired with the nearest remaining body token (Euclidean distance; first legal slot breaks ties). The latter copies a neighbor without estimating a conditional operator; it still uses body proximity.

Skill per image = 1 - sum_held ||N-pred||^2 / sum_held ||N-mean_train||^2. Table means and positive fractions are class-macro; errors sum over 32 channels. Direct 32x32 and dual kxk fold solutions were checked on one actual utility image per group: maximum discrepancy 1.11e-16, detecting transpose or ridge-scaling mistakes.

| Group | Ridge skill macro mean | Positive fraction macro | Nearest-body skill macro mean | Cbb effective rank mean | Ridge df mean | Held-body training-span fraction mean |
|---|---:|---:|---:|---:|---:|---:|
| VV_side | .13944 | .83087 | -.43763 | 3.40079 | 2.87904 | .61548 |
| VV_end | .09870 | .73015 | -.20575 | 2.21579 | 1.54440 | .26949 |
| VH_side | .13248 | .87618 | -.44823 | 3.39352 | 2.87751 | .62172 |
| VH_end | .09996 | .69644 | -.20676 | 2.18326 | 1.53546 | .27866 |

Full Cbb ranks are 7 for side, 3 for end, except the one 3-edge VH_end image with rank 2. Each LOO fold has rank at most 6/2. Ridge supplies a unique numerical answer, but it cannot identify responses outside the observed body span. The held-body span fraction is squared projection length divided by squared centered-body length, not reconstruction skill. End groups have particularly limited support (~27% on average). The ridge can exploit a modest low-dimensional within-image trend; this does not identify a transferable dense 32x32 mechanism. Region tokens also share convolutional receptive fields (layer1 RF35), so held edges are not independent chips. No shuffle causality or cross-ship prediction is claimed.

`reconstruction_summary.csv` contains class rows, macro means, and pooled rows. Its median and ratio-of-total-SSE fields are pooled even on the macro row; use the explicitly named mean fields for class-macro claims.

## Native image cases: both panels inspected

Deterministic selection: among utility queries with all four groups, choose minimum and maximum pooled four-group ridge skill, with obs_id tie order. No seed, parameter tuning, class prediction, or evaluation access. References are the highest raw VV_side-C similarity among eligible same-class and other-class bank members, separately; they are not selected classifier supports or independent trials. All shown references pass MMSI and pass exclusions. Two panels display VV and VH separately using existing geo.native, native160 crop/pad without resizing, fixed 0..255 display, with exact source paths and placements in `image_provenance.json`. These are stored 8-bit chips, not calibrated raw backscatter.

- **Low: obs4367, chemical tanker, Houston; pooled skill -.17753.** Group skills VV_side/VV_end/VH_side/VH_end = -.08032/-.36993/-.10311/-.15085. Both polarizations show a compact near-vertical bright body with textured surrounding pixels. Same-class ref23631 has a broader bright patch in VV and a compact diagonal VH response; other-class tug ref29651 also has a compact bright patch. Raw VV_side similarities .005496/.005528 and cosines .13869/.16629 slightly favor the shown other-class reference. The large crosscov norms (.145-.237) do not guarantee held-edge predictability. VV_end body training-span fraction is .303 and VH_end .183; low span coverage is measured, but does not by itself prove the cause of the negative skill.
- **High: obs20464, bulk carrier, Shanghai; pooled skill +.43634.** Group skills +.46010/+.53938/+.43092/+.33388. Both polarizations show a faint near-horizontal elongated response with brighter local points, over a dark stored chip. Same-class ref2379 has a longer, much brighter horizontal response and more visible surrounding texture; other-class general-cargo ref3536 shows a faint elongated response over a dark chip. Raw VV_side similarities .003743/.006121 and cosines .17663/.53048 favor that other-class reference. Thus even strong within-image predictability can coexist with stronger cross-class relation similarity. VV_end top singular energy .98294 and held-body span .75302 contrast with low-case .57432/.30318; this documents concentrated predictable local variation, not a named scattering mechanism.

The extreme cases are not prevalence estimates. No physical explanation (wake, sea state, saturation mechanism, or label defect) follows from these displayed intensities. Remaining gap: whether a shared learned operator can use this low-rank evidence for correct cross-ship class competition while retaining beneficial amplitude. The parent operator experiment must answer that; this diagnostic does not replace it or invalidate retained modules/modest prior gains.

## Deliverables

`run.py`, `class_kernels.csv`, `query_gaps.csv`, `reference_coverage.csv`, `image_diagnostics.csv`, `norm_distributions.csv`, `reconstruction_summary.csv`, `case_references.csv`, `image_provenance.json`, `receipt.json`, and `case_low_4367.png` / `case_high_20464.png`. No trained weights or large new feature cache.
