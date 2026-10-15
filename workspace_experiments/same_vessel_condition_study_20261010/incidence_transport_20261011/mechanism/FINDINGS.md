# Incidence transport: source-only mechanism findings

Date: 2026-10-11. Original sourcefit 1296 observations / 525 ships and heldraw
823 / 302; eight frozen full/fold bases, two seeds. No target-query readout,
neural fit, coefficient sweep, or bootstrap. Additional fitting consists only
of 80 small response-only leave-held-port-out ridge fits through the main API.

## Conclusion

The contextual incidence response fits source pair changes but does not
generalize to held vessels in this pilot. On large angle gaps its held
predictions are nearly orthogonal to actual changes, not merely too large
along a useful common direction. Response-only port exclusion does not rescue
this result. This does not establish that all same-vessel changes are noise,
that incidence has no physical effect, or that condition prediction is futile.

The sourcefit angle energy is concentrated in a few ships and has a different
class/ship mix from heldraw. That is a concrete limitation on response
estimation, not a demonstrated unique cause of failure. Uncontrolled geometry,
time, scene, state changes, and limitations of this fixed response design
remain possible explanations; this bounded diagnostic does not separate them.

## Raw pair prediction

Explained percent is 100 times the reduction of weighted squared raw delta
error relative to predicting zero. Energies sum the 32 coordinates. Classes
are equal, then ships within class, then pair directions within ship; weights
are recomputed within each subset. Entries below average the two seed metrics,
not the underlying representation coordinates.

| Arm | Sourcefit all | Sourcefit gap >=2 deg | Heldraw all | Heldraw gap >=2 deg |
| --- | ---: | ---: | ---: | ---: |
| no_condition | 7.2744 | 5.6258 | 2.6829 | 0.6383 |
| angle_shared | 0.9801 | 3.7062 | -0.2467 | -0.3293 |
| angle_context | 5.4083 | 16.8919 | -5.6316 | -19.9444 |
| angle_sham, evaluated at real angles | -0.1214 | -1.4767 | -8.4260 | -25.0455 |

The equal-capacity no_condition arm outperforms angle_context on heldraw.
Its gap>=2 result is seed-dependent (-3.1016 / +4.3783%), and near-angle results
are +2.9651 / +4.0292%. It can learn generic anchor-dependent mean correction;
this is not evidence for an incidence-specific response.

The angle_sham sourcefit rows above are NOT its actual fitting loss. Under
the actual fixed within-cell shuffled training angles, full-model explained
energy is +5.3735 / +4.6224% for seeds 31415 / 27182. Under real training angles
it is +0.5655 / -0.8083%. See sham_training_objective.csv for the actual
training data-fit error (the ridge penalty is not included in that error).

## Direction Versus Amplitude

The weighted directional cosine is <actual,predicted> divided by their pooled
weighted norms. The descriptive amplitude ratio is
<actual,predicted>/<predicted,predicted>; no held-derived scale is deployed.
Positive-projection weight counts directed pair weight with a positive dot
product, not independent successful trials or a statistical significance test.

| Context, gap >=2 deg | Direction cosine | Amplitude ratio | Positive projection weight | Prediction RMS / actual RMS |
| --- | ---: | ---: | ---: | ---: |
| Sourcefit, 31415 | 0.42734 | 1.02764 | 86.79% | 0.41585 |
| Heldraw, 31415 | -0.00675 | -0.01518 | 51.02% | 0.44423 |
| Sourcefit, 27182 | 0.39417 | 1.01199 | 85.62% | 0.38950 |
| Heldraw, 27182 | 0.01738 | 0.03779 | 48.70% | 0.45994 |

Held predictions retain substantial amplitude while useful aggregate alignment
almost vanishes. Even an unrestricted scalar refit to these same held errors
could explain only about 0.0046 / 0.0302% of energy (100*cosine^2, descriptive
ceiling, not a new fit or a validated remedy). The first seed would require a
negative multiplier. Thus simple rescaling cannot recover the trained 16.9%
large-gap gain on this held cohort.

The shared slope is different: held large-gap cosines are weakly positive,
0.05498 / 0.06216, with amplitude ratios 0.39141 / 0.44184. It has a small
aligned component but predicts too much energy to outperform zero. Do not
conflate this weak shared association with the contextual model's lost direction.

All six represented held large-gap classes have negative contextual explained
energy for both seeds. Class IDs 0, 2 and 7 have negative directional cosines
for both seeds; IDs 1, 3 and 6 have positive but insufficient alignment. Class 3
has only two ships and the largest error increase (-63.96 / -67.16%).

## Transfer And Scene Checks

For heldraw pairs whose two products both occurred in the full encoder fit,
the large-gap contextual explained percentages are -22.6008 / -21.0560%
(37 unordered pairs / 16 ships / 6 classes). Failure therefore also occurs
within familiar scenes. This does not rule out scene or other confounding.

All six fold-base heldraw large-gap contextual results are negative:
-48.80, -27.67, -37.10% for seed 31415 and -39.30, -22.27, -31.74% for 27182.
Their pooled directional cosines range from -0.0292 to +0.0335. The failure
is not confined to one full-base export or one seed.

Only response fitting excludes the evaluated port below. Base encoders,
whitening, class context and their historical selection are NOT LOPO.
Per-port predictions are pooled, then reweighted across the original held
cohort; these are not averages of port metrics.

| Response-only LOPO contextual result | Heldraw all | Heldraw gap >=2 deg | Gap>=2 cosine | Gap>=2 amplitude ratio |
| --- | ---: | ---: | ---: | ---: |
| Seed 31415 | -5.5238 | -19.8732 | 0.02831 | 0.05960 |
| Seed 27182 | -7.0131 | -23.1859 | 0.03722 | 0.07155 |

All seven ports with held large-gap pairs have negative contextual explained
energy in both seeds. New York has positive direction (cosine 0.2783 / 0.2488)
but excessive amplitude (ratios 0.3794 / 0.3032); Shanghai has reversed
alignment (-0.2748 / -0.2311) on one ship. Other small-port values remain
descriptive. LOPO no_condition large-gap results are -4.6530 / +3.9944%;
shared is -0.8665 / -0.6606%. There is no robust port-transfer rescue.

## Angle Energy Support

Energy is the fixed class/ship/pair weighted (delta_angle/5)^2. Summing it by
ship gives shares p_s; effective ships = 1/sum(p_s^2). This is a concentration
summary of scalar angle energy, not independent sample size or the effective
rank of the contextual design matrix.

| Full-base cohort | Unordered pairs / ships | Gap>=2 pairs / ships | Median absolute gap | Angle-energy effective ships | Top-five share |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sourcefit | 384 / 88 | 59 / 30 | 0.1044 deg | 6.4462 | 63.8532% |
| Heldraw | 189 / 43 | 53 / 18 | 0.8495 deg | 5.1464 | 74.2843% |

The largest source contributor is MMSI 636012630, class 5, with 35.1565% of
angle energy. Class 5 has no held gap>=2 pairs; neither does class 4, which
contains the second-largest source contributor (11.5144%). Sourcefit class 7
has no large-gap pairs but heldraw has two ships with such pairs. These are
observed support differences, not grounds for a post hoc weighting change.

Source paired endpoint angles span 30.9102--45.2184 deg; held paired endpoints
span 30.9495--45.8832 deg. An endpoint lies outside the paired source range
for 8.8819% of held class/ship/pair weight. Maximum absolute gaps are 11.7941
deg fit and 10.9805 deg held: no held gap exceeds the training maximum.
The difficulty is therefore not simply held angle differences beyond the
largest trained difference. These ranges describe paired response training,
not the deployment fallback gate's range across all known-angle fit rows.

## Original Legacy Cohort

The 384 sourcefit and 189 held unordered pair IDs exactly match the original
incidence_factor_chain cohort. The legacy B-condition32 shared-slope results
were held-all +0.2167 / +0.5053% and held-gap>=2 +3.6060 / +4.5262%.

Those old figures used the old condition representation, ship-only weighting
and chronological orientation. Current figures use whitened task32,
class/ship/pair weights and both directions. They remain valid historical
weak positives; the numerical difference is not an isolated ablation of
context conditioning. The current angle_shared and no_condition arms are
the matched quantitative comparators for the current angle_context arm.

## Verification And Artifacts

All 128 main/sidecar common sourcefit/heldraw, all/gap2 arm results agree:
maximum explained-percent difference 0; sidecar summed error /32 equals
main MSE. The legacy unordered pair IDs match exactly. Cached LOPO predictions
cover every held directed pair exactly once in each seed before pooling.

pair_prediction_metrics.csv includes per-class and per-port diagnostics;
arm_comparisons.csv contains matched-arm differences. response_leave_held_port_out.csv
retains each port; response_lopo_pooled.csv reweights the combined predictions.
angle_energy_coverage.csv and angle_energy_ship_contributions.csv retain
concentration and range evidence. Source delta caches and LOPO coefficient/
fitting-ID caches support replay. No target prediction cache was opened.

The evidence supports preserving a negative result for this specific response
learner, alongside weak shared and generic-context associations. No additional
fit, rank/weight sweep, or model promotion is justified by this diagnostic alone.
