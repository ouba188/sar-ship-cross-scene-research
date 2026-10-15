# Base-only activation + preserved historical expert

Client-local run label: 2026-10-09 (directory label retained). COMPLETE: six fixed ports, all five original64-label draws, 5% expert handoff. All six target policies froze before this run read target query outcomes. No sweep, expansion, or promotion. Historical expert scores and the complete original-reference policy remain exactly unchanged.

## Equal-port results

All OA/BA columns below are percentages; means first average five draws per port, then weight six ports equally. This difficult six-port subset must not be compared directly with full23-port means.

| arm | oa | ba | base_oa | base_ba | expert_oa | expert_ba | rescues | harms | admissions |
|---|---|---|---|---|---|---|---|---|---|
| original | 44.5655 | 45.0082 | 42.6413 | 43.2352 | 48.9740 | 48.6416 | 63.0333 | 23.2333 | 138.8333 |
| matched_baseline | 44.4722 | 45.0355 | 42.6423 | 43.2358 | 48.9740 | 48.6416 | 61.7000 | 25.5333 | 138.8333 |
| stable_only | 44.3959 | 45.2967 | 42.6771 | 43.3544 | 48.9740 | 48.6416 | 60.9000 | 26.2333 | 138.8333 |
| joint | 44.2589 | 45.1353 | 42.4421 | 43.1727 | 48.9740 | 48.6416 | 61.9667 | 23.9667 | 138.8333 |

## Paired changes

All deltas are percentage points.

| arm | control | delta_oa_pp | delta_ba_pp | delta_base_ba_pp | delta_expert_ba_pp | delta_handoff_ba_pp |
|---|---|---|---|---|---|---|
| matched_baseline | original | -0.0933 | 0.0274 | 0.0007 | 0.0000 | 0.0267 |
| stable_only | matched_baseline | -0.0763 | 0.2612 | 0.1186 | 0.0000 | 0.1425 |
| joint | matched_baseline | -0.2133 | 0.0997 | -0.0631 | 0.0000 | 0.1629 |
| joint | stable_only | -0.1370 | -0.1615 | -0.1818 | 0.0000 | 0.0203 |
| stable_only | original | -0.1695 | 0.2885 | 0.1193 | 0.0000 | 0.1692 |
| joint | original | -0.3066 | 0.1271 | -0.0625 | 0.0000 | 0.1896 |

## Direct expert-preservation comparison

Read-only comparison with the completed protected_drift_activation_20261008 cached metrics, restricted to these identical six ports and five draws. Corresponding base OA/BA, query counts and handoff quotas match exactly. Both recipes fit a compatible source gate; therefore this tests preserved-expert-plus-compatible-gate versus changed-expert-plus-compatible-gate, not expert replacement under a single fixed gate. No new training or target selection was performed.

| Arm | Both changed OA % | Preserved OA % | OA delta pp | Both changed BA % | Preserved BA % | BA delta pp | Expert BA delta pp |
|---|---:|---:|---:|---:|---:|---:|---:|
| matched_baseline | 44.8832 | 44.4722 | -0.4110 | 44.9603 | 45.0355 | +0.0753 | +0.0948 |
| stable_only | 44.9533 | 44.3959 | -0.5573 | 45.2556 | 45.2967 | +0.0412 | +0.1815 |
| joint | 44.7602 | 44.2589 | -0.5013 | 45.0929 | 45.1353 | +0.0424 | +0.1439 |

Preservation raises final BA only slightly relative to the corresponding both-changed recipes (about0.04 pp for stable-only/joint), while lowering OA by about0.50-0.56 pp. It preserves higher standalone expert BA, but does NOT establish a better overall handoff system. The stronger preserved expert need not be more helpful on the particular5% disagreements selected. This direct result weakens, rather than confirms, a broad claim that preserving the historical expert is the missing route to a material gain.


## Source-only bounded diagnosis

These diagnostic gates use three blocked episode folds (episode index modulo3), excluding each held episode from gate fitting. Upstream source geometry, encoder and precomputed head predictions stay fixed; this is gate-only crossfit, NOT a fully nested end-to-end validation. No diagnostic selected a recipe.

| arm | oa | ba | base_ba | expert_ba | rescues | harms | admissions |
|---|---|---|---|---|---|---|---|
| joint | 61.5866 | 55.0645 | 53.0257 | 57.1129 | 47.4545 | 11.5455 | 73.2955 |
| matched_baseline | 57.5012 | 51.4504 | 49.2092 | 57.1129 | 48.8636 | 10.1818 | 73.2955 |
| stable_only | 61.5719 | 55.2332 | 53.2658 | 57.1129 | 46.8939 | 12.2348 | 73.2955 |
- stable_only: target base BA +0.1186 pp; additional handoff contribution +0.1425 pp; final BA +0.2612 pp. Source base BA +4.0567 pp; gate-crossfit final BA +3.7828 pp. Target ports up/down: 3/3.
- joint: target base BA -0.0631 pp; additional handoff contribution +0.1629 pp; final BA +0.0997 pp. Source base BA +3.8165 pp; gate-crossfit final BA +3.6141 pp. Target ports up/down: 3/3.

Bounded finding: stable-only has positive mean target BA in all five draws but only three of six ports improve, and mean OA falls. Its much larger source gain does not transfer at the same magnitude. Because source candidate geometry used inner-held labels, that source gain cannot be treated as an unbiased estimate of unseen-port gain. Joint versus stable-only reduces base BA on both the source diagnostic and target readout: the additional attenuation harms this base readout, while keeping the expert fixed cannot restore all of the lost benefit. This does not prove that Qd is purely useful or purely nuisance information. No implementation defect is established, and the remaining split between source geometry optimism and actual port heterogeneity is unresolved.

The additive final-minus-base decomposition identifies where measured gains/losses enter the pipeline, not a causal attribution. An unchanged expert can still have a different marginal handoff benefit when base decisions and eligibility change. Source crossfit distinguishes a source-visible loss from a target-only reversal; it does not identify a unique latent mechanism.

Implementation checks: all132 source episodes reproduce the original protected-activation base OA/BA; all90 cached target base score arrays independently replay exactly; all30 target expert score arrays and original policies are exactly unchanged. Row IDs, original64 supports, query masks and quotas match. Five focused tests cover explicit views, independent primal ridge equivalence, original binary gate objective, expert nonmutation/5% routing and corrupted support/row alignment. Thus no observed score-reconstruction, wrong-row, changed-expert or budget error explains any negative result.

## Pipeline and boundaries

1. Reuse source-discovered candidate Qs and protected drift Qd from the completed20261008 run; these are candidate stable directions, not proven causal invariants.
2. Matched baseline refits only the ordinary base source head. Stable-only appends source-standardized Qs coordinates to that head and64-label residual features. Joint also attenuates16 orthogonal drift directions by0.5. No expert source head, expert kernel or expert support correction is changed.
3. Reuse exact saved target base scores; rebuild source base heads using original purged train rows. Refit each compatible four-confidence gate using source eligible informative rescue/harm disagreements, equal episode/class weighting and LR C1.
4. Among query disagreements whose expert class is present in the64 supports and whose gate score is positive, hand off at most floor(0.05*Nquery), using row-ID ties.
5. Qs/Qd/PCA/scales and encoder use all outer-source data including inner-held source labels. Only source readout fits are inner-held. Existing data have repeatedly supported development; neither the six ports nor overlapping draws constitute independent confirmation.

The historical reference retains old base and old gate. The matched baseline intentionally includes base refitting, so historical-reference improvement alone is not the stable-module effect. No same-dimensional generic extra branch is tested here, so even positive stable activation does not establish exclusive stability causation. Main model is not replaced.

## Every port

| port | arm | oa | ba | base_oa | base_ba | expert_oa | expert_ba | admissions |
|---|---|---|---|---|---|---|---|---|
| Antwerp-Bruges | joint | 73.0170 | 76.4755 | 69.0511 | 72.8510 | 81.4112 | 83.7543 | 41.0000 |
| Antwerp-Bruges | matched_baseline | 72.2871 | 75.0619 | 68.3455 | 71.7285 | 81.4112 | 83.7543 | 41.0000 |
| Antwerp-Bruges | original | 72.3358 | 74.9009 | 68.3455 | 71.7285 | 81.4112 | 83.7543 | 41.0000 |
| Antwerp-Bruges | stable_only | 73.5280 | 77.3522 | 69.7567 | 73.8555 | 81.4112 | 83.7543 | 41.0000 |
| Jebel Ali | joint | 37.0251 | 39.3296 | 35.7162 | 38.1809 | 39.9956 | 41.0595 | 226.0000 |
| Jebel Ali | matched_baseline | 37.7788 | 39.8344 | 36.5051 | 38.7100 | 39.9956 | 41.0595 | 226.0000 |
| Jebel Ali | original | 37.7655 | 39.8180 | 36.5051 | 38.7100 | 39.9956 | 41.0595 | 226.0000 |
| Jebel Ali | stable_only | 36.8709 | 39.1753 | 35.8572 | 38.3339 | 39.9956 | 41.0595 | 226.0000 |
| Port Klang | joint | 41.6781 | 36.5189 | 39.9405 | 35.2793 | 48.5391 | 39.4761 | 168.0000 |
| Port Klang | matched_baseline | 41.6007 | 37.2094 | 39.9226 | 36.0993 | 48.5391 | 39.4761 | 168.0000 |
| Port Klang | original | 41.5055 | 36.9522 | 39.9167 | 36.0954 | 48.5391 | 39.4761 | 168.0000 |
| Port Klang | stable_only | 41.9518 | 36.5218 | 40.3392 | 35.3209 | 48.5391 | 39.4761 | 168.0000 |
| Santos | joint | 39.1217 | 42.5168 | 37.6412 | 39.2892 | 41.7817 | 45.8384 | 79.0000 |
| Santos | matched_baseline | 39.6236 | 42.3145 | 38.1932 | 39.6867 | 41.7817 | 45.8384 | 79.0000 |
| Santos | original | 39.7491 | 42.3542 | 38.1932 | 39.6867 | 41.7817 | 45.8384 | 79.0000 |
| Santos | stable_only | 39.4354 | 43.1947 | 37.9674 | 39.6199 | 41.7817 | 45.8384 | 79.0000 |
| Singapore | joint | 34.6449 | 35.7244 | 34.0173 | 34.7367 | 37.7401 | 39.7022 | 259.0000 |
| Singapore | matched_baseline | 34.4947 | 36.5073 | 34.1020 | 35.7334 | 37.7401 | 39.7022 | 259.0000 |
| Singapore | original | 34.9220 | 36.7957 | 34.1020 | 35.7334 | 37.7401 | 39.7022 | 259.0000 |
| Singapore | stable_only | 34.5563 | 35.8817 | 33.9557 | 34.9370 | 37.7401 | 39.7022 | 259.0000 |
| Tanger Med | joint | 40.0666 | 40.2465 | 38.2862 | 38.6989 | 44.3760 | 42.0192 | 60.0000 |
| Tanger Med | matched_baseline | 41.0483 | 39.2858 | 38.7854 | 37.4570 | 44.3760 | 42.0192 | 60.0000 |
| Tanger Med | original | 41.1148 | 39.2282 | 38.7854 | 37.4570 | 44.3760 | 42.0192 | 60.0000 |
| Tanger Med | stable_only | 40.0333 | 39.6546 | 38.1864 | 38.0595 | 44.3760 | 42.0192 | 60.0000 |

## Every draw and port

| port | draw | arm | oa | ba | base_oa | base_ba | expert_oa | expert_ba | admissions |
|---|---|---|---|---|---|---|---|---|---|
| Antwerp-Bruges | 0 | joint | 74.2092 | 78.9333 | 70.4380 | 75.3643 | 81.2652 | 84.9679 | 41 |
| Antwerp-Bruges | 0 | matched_baseline | 71.7762 | 76.1655 | 67.7616 | 73.0722 | 81.2652 | 84.9679 | 41 |
| Antwerp-Bruges | 0 | original | 71.7762 | 75.9439 | 67.7616 | 73.0722 | 81.2652 | 84.9679 | 41 |
| Antwerp-Bruges | 0 | stable_only | 74.0876 | 79.3995 | 70.5596 | 75.9926 | 81.2652 | 84.9679 | 41 |
| Antwerp-Bruges | 1 | joint | 72.7494 | 76.2445 | 68.9781 | 72.5387 | 81.6302 | 83.7080 | 41 |
| Antwerp-Bruges | 1 | matched_baseline | 73.1144 | 75.0721 | 69.0998 | 71.3835 | 81.6302 | 83.7080 | 41 |
| Antwerp-Bruges | 1 | original | 72.8710 | 74.4639 | 69.0998 | 71.3835 | 81.6302 | 83.7080 | 41 |
| Antwerp-Bruges | 1 | stable_only | 73.4793 | 77.1655 | 69.9513 | 73.6478 | 81.6302 | 83.7080 | 41 |
| Antwerp-Bruges | 2 | joint | 70.6813 | 74.3515 | 66.1800 | 70.1067 | 81.5085 | 84.2337 | 41 |
| Antwerp-Bruges | 2 | matched_baseline | 71.7762 | 75.2349 | 67.6399 | 71.8923 | 81.5085 | 84.2337 | 41 |
| Antwerp-Bruges | 2 | original | 72.0195 | 75.4794 | 67.6399 | 71.8923 | 81.5085 | 84.2337 | 41 |
| Antwerp-Bruges | 2 | stable_only | 71.7762 | 75.9725 | 67.7616 | 72.0220 | 81.5085 | 84.2337 | 41 |
| Antwerp-Bruges | 3 | joint | 72.5061 | 76.6205 | 68.9781 | 73.0754 | 80.5353 | 83.1113 | 41 |
| Antwerp-Bruges | 3 | matched_baseline | 70.6813 | 74.4523 | 66.9100 | 70.5731 | 80.5353 | 83.1113 | 41 |
| Antwerp-Bruges | 3 | original | 70.6813 | 73.9878 | 66.9100 | 70.5731 | 80.5353 | 83.1113 | 41 |
| Antwerp-Bruges | 3 | stable_only | 72.7494 | 77.1005 | 69.2214 | 73.4394 | 80.5353 | 83.1113 | 41 |
| Antwerp-Bruges | 4 | joint | 74.9392 | 76.2275 | 70.6813 | 73.1699 | 82.1168 | 82.7507 | 41 |
| Antwerp-Bruges | 4 | matched_baseline | 74.0876 | 74.3849 | 70.3163 | 71.7215 | 82.1168 | 82.7507 | 41 |
| Antwerp-Bruges | 4 | original | 74.3309 | 74.6297 | 70.3163 | 71.7215 | 82.1168 | 82.7507 | 41 |
| Antwerp-Bruges | 4 | stable_only | 75.5474 | 77.1232 | 71.2895 | 74.1760 | 82.1168 | 82.7507 | 41 |
| Jebel Ali | 0 | joint | 37.6598 | 40.1133 | 35.9409 | 38.5351 | 38.5853 | 41.6354 | 226 |
| Jebel Ali | 0 | matched_baseline | 37.8140 | 39.7797 | 36.2935 | 38.5599 | 38.5853 | 41.6354 | 226 |
| Jebel Ali | 0 | original | 37.7479 | 39.7242 | 36.2935 | 38.5599 | 38.5853 | 41.6354 | 226 |
| Jebel Ali | 0 | stable_only | 37.4174 | 39.5943 | 36.0511 | 38.7233 | 38.5853 | 41.6354 | 226 |
| Jebel Ali | 1 | joint | 33.5831 | 37.7307 | 33.6271 | 37.4316 | 38.0564 | 39.5919 | 226 |
| Jebel Ali | 1 | matched_baseline | 35.4121 | 39.3282 | 35.5223 | 38.7452 | 38.0564 | 39.5919 | 226 |
| Jebel Ali | 1 | original | 36.0511 | 39.7521 | 35.5223 | 38.7452 | 38.0564 | 39.5919 | 226 |
| Jebel Ali | 1 | stable_only | 33.2746 | 37.4498 | 33.5610 | 37.3347 | 38.0564 | 39.5919 | 226 |
| Jebel Ali | 2 | joint | 39.4888 | 39.7369 | 37.8140 | 38.4776 | 43.0366 | 41.6826 | 226 |
| Jebel Ali | 2 | matched_baseline | 38.9819 | 40.1253 | 37.2631 | 38.9175 | 43.0366 | 41.6826 | 226 |
| Jebel Ali | 2 | original | 38.9379 | 40.2934 | 37.2631 | 38.9175 | 43.0366 | 41.6826 | 226 |
| Jebel Ali | 2 | stable_only | 39.5328 | 39.3847 | 38.3208 | 38.6824 | 43.0366 | 41.6826 | 226 |
| Jebel Ali | 3 | joint | 37.9022 | 39.2724 | 36.6461 | 38.4155 | 39.8193 | 41.5285 | 226 |
| Jebel Ali | 3 | matched_baseline | 38.5853 | 39.3333 | 37.4835 | 38.5719 | 39.8193 | 41.5285 | 226 |
| Jebel Ali | 3 | original | 38.4751 | 39.1026 | 37.4835 | 38.5719 | 39.8193 | 41.5285 | 226 |
| Jebel Ali | 3 | stable_only | 37.4835 | 39.1999 | 36.6461 | 38.6162 | 39.8193 | 41.5285 | 226 |
| Jebel Ali | 4 | joint | 36.4918 | 39.7944 | 34.5527 | 38.0449 | 40.4804 | 40.8593 | 226 |
| Jebel Ali | 4 | matched_baseline | 38.1005 | 40.6054 | 35.9630 | 38.7553 | 40.4804 | 40.8593 | 226 |
| Jebel Ali | 4 | original | 37.6157 | 40.2177 | 35.9630 | 38.7553 | 40.4804 | 40.8593 | 226 |
| Jebel Ali | 4 | stable_only | 36.6461 | 40.2480 | 34.7069 | 38.3127 | 40.4804 | 40.8593 | 226 |
| Port Klang | 0 | joint | 42.4278 | 35.3250 | 41.1782 | 34.3731 | 46.8908 | 37.8687 | 168 |
| Port Klang | 0 | matched_baseline | 41.2972 | 36.4903 | 40.1964 | 35.7538 | 46.8908 | 37.8687 | 168 |
| Port Klang | 0 | original | 41.3567 | 36.2891 | 40.1666 | 35.7342 | 46.8908 | 37.8687 | 168 |
| Port Klang | 0 | stable_only | 42.6956 | 35.4543 | 41.4758 | 34.5448 | 46.8908 | 37.8687 | 168 |
| Port Klang | 1 | joint | 36.9533 | 34.9919 | 35.2574 | 33.6039 | 46.8313 | 39.6343 | 168 |
| Port Klang | 1 | matched_baseline | 37.0128 | 35.3074 | 35.7037 | 34.2536 | 46.8313 | 39.6343 | 168 |
| Port Klang | 1 | original | 36.7153 | 35.0439 | 35.7037 | 34.2536 | 46.8313 | 39.6343 | 168 |
| Port Klang | 1 | stable_only | 37.1616 | 34.7986 | 35.7037 | 33.6695 | 46.8313 | 39.6343 | 168 |
| Port Klang | 2 | joint | 40.4641 | 38.1760 | 39.2443 | 37.5260 | 48.0809 | 40.6608 | 168 |
| Port Klang | 2 | matched_baseline | 41.4758 | 38.4899 | 40.1964 | 37.4290 | 48.0809 | 40.6608 | 168 |
| Port Klang | 2 | original | 41.3270 | 38.4366 | 40.1964 | 37.4290 | 48.0809 | 40.6608 | 168 |
| Port Klang | 2 | stable_only | 40.5832 | 38.2602 | 39.6013 | 37.6197 | 48.0809 | 40.6608 | 168 |
| Port Klang | 3 | joint | 44.2130 | 36.3437 | 41.2675 | 34.1297 | 51.5025 | 39.6383 | 168 |
| Port Klang | 3 | matched_baseline | 44.0940 | 37.1651 | 41.0592 | 35.6077 | 51.5025 | 39.6383 | 168 |
| Port Klang | 3 | original | 43.7370 | 36.4048 | 41.0592 | 35.6077 | 51.5025 | 39.6383 | 168 |
| Port Klang | 3 | stable_only | 44.6296 | 36.1966 | 41.7138 | 34.0322 | 51.5025 | 39.6383 | 168 |
| Port Klang | 4 | joint | 44.3320 | 37.7580 | 42.7551 | 36.7636 | 49.3901 | 39.5784 | 168 |
| Port Klang | 4 | matched_baseline | 44.1238 | 38.5941 | 42.4576 | 37.4524 | 49.3901 | 39.5784 | 168 |
| Port Klang | 4 | original | 44.3916 | 38.5866 | 42.4576 | 37.4524 | 49.3901 | 39.5784 | 168 |
| Port Klang | 4 | stable_only | 44.6891 | 37.8990 | 43.2014 | 36.7383 | 49.3901 | 39.5784 | 168 |
| Santos | 0 | joint | 42.5345 | 43.6006 | 40.7152 | 40.0008 | 46.7378 | 47.6398 | 79 |
| Santos | 0 | matched_baseline | 42.6600 | 43.1596 | 40.6524 | 38.9004 | 46.7378 | 47.6398 | 79 |
| Santos | 0 | original | 42.8482 | 42.9625 | 40.6524 | 38.9004 | 46.7378 | 47.6398 | 79 |
| Santos | 0 | stable_only | 43.0991 | 44.1662 | 41.4680 | 40.3448 | 46.7378 | 47.6398 | 79 |
| Santos | 1 | joint | 38.5822 | 37.5599 | 37.3275 | 35.0484 | 42.9109 | 42.2685 | 79 |
| Santos | 1 | matched_baseline | 39.2723 | 38.8759 | 37.5157 | 36.8773 | 42.9109 | 42.2685 | 79 |
| Santos | 1 | original | 39.1468 | 38.7194 | 37.5157 | 36.8773 | 42.9109 | 42.2685 | 79 |
| Santos | 1 | stable_only | 39.1468 | 38.1710 | 37.8921 | 35.7347 | 42.9109 | 42.2685 | 79 |
| Santos | 2 | joint | 36.1355 | 40.8675 | 35.2572 | 38.2967 | 37.8921 | 44.8969 | 79 |
| Santos | 2 | matched_baseline | 36.8256 | 39.6713 | 36.6374 | 37.8718 | 37.8921 | 44.8969 | 79 |
| Santos | 2 | original | 36.9511 | 39.6469 | 36.6374 | 37.8718 | 37.8921 | 44.8969 | 79 |
| Santos | 2 | stable_only | 36.3237 | 42.0298 | 35.1945 | 38.1078 | 37.8921 | 44.8969 | 79 |
| Santos | 3 | joint | 39.6487 | 43.7597 | 38.2685 | 39.8106 | 40.8407 | 45.8963 | 79 |
| Santos | 3 | matched_baseline | 41.2171 | 43.8892 | 39.8369 | 40.8722 | 40.8407 | 45.8963 | 79 |
| Santos | 3 | original | 41.5307 | 44.3622 | 39.8369 | 40.8722 | 40.8407 | 45.8963 | 79 |
| Santos | 3 | stable_only | 40.0251 | 44.5140 | 38.6449 | 40.4311 | 40.8407 | 45.8963 | 79 |
| Santos | 4 | joint | 38.7077 | 46.7962 | 36.6374 | 43.2893 | 40.5270 | 48.4903 | 79 |
| Santos | 4 | matched_baseline | 38.1430 | 45.9766 | 36.3237 | 43.9114 | 40.5270 | 48.4903 | 79 |
| Santos | 4 | original | 38.2685 | 46.0798 | 36.3237 | 43.9114 | 40.5270 | 48.4903 | 79 |
| Santos | 4 | stable_only | 38.5822 | 47.0925 | 36.6374 | 43.4811 | 40.5270 | 48.4903 | 79 |
| Singapore | 0 | joint | 36.3619 | 35.4500 | 35.6304 | 34.3870 | 39.3455 | 39.7807 | 259 |
| Singapore | 0 | matched_baseline | 35.8807 | 35.7432 | 35.5534 | 35.2980 | 39.3455 | 39.7807 | 259 |
| Singapore | 0 | original | 36.6506 | 36.5822 | 35.5534 | 35.2980 | 39.3455 | 39.7807 | 259 |
| Singapore | 0 | stable_only | 36.3811 | 36.1652 | 35.5149 | 34.7885 | 39.3455 | 39.7807 | 259 |
| Singapore | 1 | joint | 31.2031 | 35.9777 | 30.8951 | 35.1831 | 35.1684 | 36.9166 | 259 |
| Singapore | 1 | matched_baseline | 32.3773 | 36.1862 | 32.4350 | 35.2868 | 35.1684 | 36.9166 | 259 |
| Singapore | 1 | original | 33.0125 | 36.5807 | 32.4350 | 35.2868 | 35.1684 | 36.9166 | 259 |
| Singapore | 1 | stable_only | 30.9721 | 35.6051 | 30.6641 | 34.8545 | 35.1684 | 36.9166 | 259 |
| Singapore | 2 | joint | 34.5717 | 33.7692 | 33.7825 | 32.9672 | 38.9605 | 40.0930 | 259 |
| Singapore | 2 | matched_baseline | 34.2445 | 34.9190 | 33.6670 | 34.2861 | 38.9605 | 40.0930 | 259 |
| Singapore | 2 | original | 34.3407 | 34.7939 | 33.6670 | 34.2861 | 38.9605 | 40.0930 | 259 |
| Singapore | 2 | stable_only | 34.3792 | 33.8837 | 33.5900 | 33.0313 | 38.9605 | 40.0930 | 259 |
| Singapore | 3 | joint | 36.1886 | 38.0293 | 35.9384 | 36.5914 | 37.7863 | 41.8800 | 259 |
| Singapore | 3 | matched_baseline | 35.4572 | 38.8116 | 35.1492 | 37.3836 | 37.7863 | 41.8800 | 259 |
| Singapore | 3 | original | 35.7844 | 38.8295 | 35.1492 | 37.3836 | 37.7863 | 41.8800 | 259 |
| Singapore | 3 | stable_only | 36.2464 | 38.0887 | 36.1501 | 37.2777 | 37.7863 | 41.8800 | 259 |
| Singapore | 4 | joint | 34.8989 | 35.3959 | 33.8402 | 34.5550 | 37.4398 | 39.8407 | 259 |
| Singapore | 4 | matched_baseline | 34.5140 | 36.8766 | 33.7055 | 36.4127 | 37.4398 | 39.8407 | 259 |
| Singapore | 4 | original | 34.8219 | 37.1922 | 33.7055 | 36.4127 | 37.4398 | 39.8407 | 259 |
| Singapore | 4 | stable_only | 34.8027 | 35.6659 | 33.8595 | 34.7331 | 37.4398 | 39.8407 | 259 |
| Tanger Med | 0 | joint | 42.1797 | 38.4849 | 40.4326 | 37.2293 | 43.8436 | 38.7775 | 60 |
| Tanger Med | 0 | matched_baseline | 42.0133 | 38.9401 | 39.7671 | 37.4133 | 43.8436 | 38.7775 | 60 |
| Tanger Med | 0 | original | 41.5973 | 38.4741 | 39.7671 | 37.4133 | 43.8436 | 38.7775 | 60 |
| Tanger Med | 0 | stable_only | 42.8453 | 39.2465 | 40.8486 | 37.6267 | 43.8436 | 38.7775 | 60 |
| Tanger Med | 1 | joint | 41.5141 | 43.6428 | 39.6007 | 42.4136 | 44.2596 | 43.2212 | 60 |
| Tanger Med | 1 | matched_baseline | 40.7654 | 40.3619 | 38.7687 | 39.0053 | 44.2596 | 43.2212 | 60 |
| Tanger Med | 1 | original | 41.1814 | 40.5892 | 38.7687 | 39.0053 | 44.2596 | 43.2212 | 60 |
| Tanger Med | 1 | stable_only | 40.8486 | 43.2388 | 39.1847 | 42.1428 | 44.2596 | 43.2212 | 60 |
| Tanger Med | 2 | joint | 36.6889 | 41.5439 | 35.3577 | 40.2872 | 41.8469 | 43.4208 | 60 |
| Tanger Med | 2 | matched_baseline | 37.5208 | 41.3169 | 36.0233 | 39.8587 | 41.8469 | 43.4208 | 60 |
| Tanger Med | 2 | original | 37.8536 | 41.5010 | 36.0233 | 39.8587 | 41.8469 | 43.4208 | 60 |
| Tanger Med | 2 | stable_only | 36.4393 | 40.7919 | 35.1082 | 39.5198 | 41.8469 | 43.4208 | 60 |
| Tanger Med | 3 | joint | 40.8486 | 38.7988 | 38.0200 | 36.0968 | 46.9218 | 42.1445 | 60 |
| Tanger Med | 3 | matched_baseline | 43.1780 | 38.4295 | 40.2662 | 35.7363 | 46.9218 | 42.1445 | 60 |
| Tanger Med | 3 | original | 43.3444 | 38.2827 | 40.2662 | 35.7363 | 46.9218 | 42.1445 | 60 |
| Tanger Med | 3 | stable_only | 40.8486 | 38.5281 | 38.1032 | 36.0644 | 46.9218 | 42.1445 | 60 |
| Tanger Med | 4 | joint | 39.1015 | 38.7619 | 38.0200 | 37.4677 | 45.0083 | 42.5318 | 60 |
| Tanger Med | 4 | matched_baseline | 41.7637 | 37.3805 | 39.1015 | 35.2712 | 45.0083 | 42.5318 | 60 |
| Tanger Med | 4 | original | 41.5973 | 37.2938 | 39.1015 | 35.2712 | 45.0083 | 42.5318 | 60 |
| Tanger Med | 4 | stable_only | 39.1847 | 36.4677 | 37.6872 | 34.9436 | 45.0083 | 42.5318 | 60 |

## Files

- metrics.csv:120 port/draw/arm records, including OA, BA, base/expert metrics, rescues/harms and quotas.
- port_means.csv, equal_port_means.csv, draw_means.csv, deltas.csv: grouped results and paired deltas.
- source_crossfit_metrics.csv:396 held-source arm records; source_fit_metrics.csv is fitting-sample gate diagnostic only.
- preservation_vs_both_changed_{draws,ports,means}.csv: direct matched comparison to prior both-changed cached recipes.
- ports/*/policies.npz: frozen target policies; one shared historical expert score array per port.
- ports/*/*_source_scores.npz: source episodes, offsets, labels, base/expert scores and gate inputs.
- PROTOCOL.json, ALL_FROZEN.json, COMPLETE.json, DIAGNOSIS.json: scope and completion records.
