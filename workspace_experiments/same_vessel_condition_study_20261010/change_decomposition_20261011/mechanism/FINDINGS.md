# Source-Only Mechanism Findings

Four full exports, two seeds, no fold diagnostics, no neural training or target evaluation. Probes use the existing ridge=1 fit_probe/score_probe and identical class/ship/acquisition/chip-balanced sourcefit weights. All following summary values are arithmetic means over the two seeds on test_source.

There is modest observed change separation, with a reconstruction tradeoff and seed-dependent class readability. The result does not establish clean disentanglement, fixed physical axes, or a unique paired downstream benefit.

| Held-source metric | Class-only | Paired |
| --- | ---: | ---: |
| Content probe BA (%) | 38.624618 | 38.994224 |
| Content probe OA (%) | 43.317132 | 43.803159 |
| Residual probe BA (%) | 11.943938 | 17.741319 |
| Original z probe BA (%) | 38.712617 | 38.712617 |
| Selected content head BA (%) | 38.719315 | 38.994224 |
| Residual delta explained fraction, all | 0.855812 | 0.864227 |
| Residual delta explained fraction, cross-port | 0.857408 | 0.873025 |
| Content delta energy / original delta energy, all | 0.153897 | 0.133540 |
| Content delta energy / original delta energy, cross-port | 0.155149 | 0.134411 |
| Full reconstruction delta error ratio, all | 0.005578 | 0.029505 |
| Full reconstruction delta error ratio, cross-port | 0.005621 | 0.025751 |
| Full pointwise reconstruction error / original energy | 0.005357 | 0.031258 |
| Matched content same-vessel / other-vessel distance | 1.032672 | 1.020257 |

The paired content probe gains 0.369606 BA points on average, but the seed differences are +1.472313 (31415) and -0.733101 (27182). Paired BA exceeds original z by 0.281607 points on average. Its selected trained head has identical held OA/BA to its independent probe in both seeds; these diagnostics do not reveal a content-probe versus content-head accuracy mismatch. This does not test the separate episodic correction objective or adapted decision rule.

Both seeds reduce held content delta energy and improve residual delta explained fraction relative to class-only. Content delta energy falls approximately 13.23% for all distinct-product pairs and 13.37% for cross-port pairs. The normalized reconstruction error increases in both seeds, especially seed 27182. Residual class readability is not expected to vanish and must not be interpreted as a nuisance-only residual.

## Acquisition-Mean Input

The selected model was evaluated on source acquisition-mean z and compared against acquisition means of the exported chip components. Both source cohorts have exactly one chip per MMSI/product/port acquisition: 1,296 fit rows/acquisitions and 823 held rows/acquisitions. Consequently, the potential nonlinear Jensen discrepancy is not material here. Normalized residual Jensen delta RMS is zero on sourcefit and cross-port held pairs; across all held pairs it is at most 8.33e-9 over the four exports. Training-aligned and chip-aggregated residual explained fractions agree to displayed precision. This conclusion is specific to these cohorts, not to future multi-chip acquisitions.

## Mechanical Partition

Decoder components are orthogonal. With e = delta_content + delta_residual - delta_z, residual delta error energy equals content delta energy + ||e||^2 - 2 delta_content dot e. Under exact reconstruction, residual explained fraction is mechanically 1 minus the content delta energy ratio. Thus these two metrics are not independent evidence of separation. Reconstruction is approximate after training, and its measured error must accompany either metric. The observed normalized content/residual delta inner products are about 1e-9 or smaller.

## Actual Coverage

- Sourcefit: 1,296 chips, 525 ships. All pairs: 2,281 pairs / 269 ships / 8 classes; cross-port: 585 pairs / 133 ships / 7 classes.
- Held source: 823 chips, 302 ships. All pairs: 1,492 pairs / 200 ships / 8 classes; cross-port: 479 pairs / 116 ships / 7 classes.
- Class 7 has no cross-port pair coverage in either cohort. Held class 6 cross-port coverage is only one pair from one ship. These classes must not be described as having demonstrated cross-port robustness.
- Existing independent matched cohort: 359 matches, 233 anchors, 86 ships, 6 classes. Content same/other distance ratios improve slightly in both seeds but remain above one. This is limited identity discrimination evidence, not proof of causal separation.

Per-class recalls and pair metrics are retained in class_recall_held_summary.csv and paired_delta_held_summary.csv; coverage.csv contains counts without duplicating cohorts across seeds or methods.

## Core Design Review

Read-only inspection of change_decomposition.py and run_change_decomposition.py found no concrete implementation defect within the reviewed source decomposition and readout paths. In particular, acquisition-mean training input versus mean chip output is a legitimate distinction whose measured effect here is negligible. No core files were edited and no additional training was run. The diagnostic does not independently certify every correction or deployment path; those replays belong to the main task.
