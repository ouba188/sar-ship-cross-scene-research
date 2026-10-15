# F0921 matched detector recrop, 2026-10-07

## Result

Correcting the target input pipeline materially improves the existing model.
All values are percentages, averaged over the same five support draws.

| Arm | Old OA | Recrop OA | OA change, pp | Old BA | Recrop BA | BA change, pp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Source-only base | 17.920356 | 24.207575 | +6.287219 | 18.846647 | 20.314242 | +1.467595 |
| Base with global64 labels | 21.433259 | 28.937087 | +7.503828 | 20.441980 | 24.901788 | +4.459808 |
| Full model, 5% handoff | 21.834150 | 29.633543 | +7.799392 | 20.839997 | 25.105149 | +4.265153 |

Full-model recrop-minus-old BA by draw: +3.091329, +4.271193,
+4.856053, +4.570440, +4.536748 pp. Corresponding OA differences:
+4.917156, +6.589427, +7.066640, +11.360745, +9.062994 pp.
These are overlapping support draws, not independent datasets.

Within the NEW input pipeline, handoff versus adapted base contributes
**+0.696455 pp OA and +0.203362 pp BA**. OA improves in all five draws;
BA improves in only three, with draw0=-0.023230 and draw3=-0.210043 pp.
The old handoff increment was +0.400891 pp OA / +0.398017 pp BA.
Thus handoff BA benefit did not strengthen. Do not attribute the large input
correction gain to a new expert module or claim uniform class improvement.

## What Was Fixed

The old producer `E:/Hermes/scripts/f0921_crop.py` enumerates DET records,
then uses their positional indices to retrieve ALIGNED geometries/attributes.
Its manifest combines ALIGNED MMSI/class with coordinates from another cached
metadata stream. `fix_relabel_points.py` can omit features, so positional
correspondence between those trees is not reliable. The ALIGNED tree is no
longer present. Current DET has both raw paired rasters and detection geometry.

Concrete observed mismatch: old chip row32046, scene
`SAR_Ship_Chip_20210902_UTM19N_EPSG32619_10m`, records row_in_shp7 and
MMSI563110100. Current DET feature7 has MMSI538008990; MMSI563110100 is
feature6. Old stored coordinates agree with feature7, not feature6.
The exact historical scene-specific deletion/reordering is not reconstructed.
This establishes an unsafe positional join, not that every old chip label is wrong.

The current recrop joins within scene by unique MMSI. For repeated MMSIs,
unique rounded lon/lat agreement is a heuristic fallback, not a verified
observation ID, because old coordinates may belong to another row.
Unresolved identities retain their old inputs. No class label, AIS length,
query correctness, or model score is used to choose a crop.

Detector provenance is the existing single-class Ship YOLO-OBB
`E:/Docms/SARlabelimg/550k.pt`, with conf0.25, window/imgsz1024 and maximum
point-to-detection distance160px, documented in the existing DET run config.
The detector was NOT rerun or retrained. Its nearest-detection association
is not treated as new ground-truth localization.

Use DET SHP geometry, not the unchanged lon/lat attributes, for pixel centers.
Native rasters are read through bounded GDAL windows, never full-scene arrays.
The source producer `E:/临时会话/build_v5_context_chips.py` defines:

```text
L = ceil(max(1, detector_longside))
S = 8 * ceil(max(L, min(128, max(64, ceil(1.8*L)))) / 8)
```

128 is a soft cap; retain larger targets. Source uses rounded box extents,
whereas this target has only OBB longside, so sizing is analogous, not exact
source box reconstruction. Window origins use round(center-S/2) and edge
clamping. Dynamic side median64, p90=88, max352 pixels.

Keep the target's per-band per-chip 2/98 percentile stretch, followed by PIL
bilinear128. Source chips copied existing 8-bit tiles without per-chip stretch;
their upstream raw-to-8-bit recipe is still unresolved. Therefore this is NOT
complete radiometric harmonization or a pure scale-only ablation: changing the
crop window also changes which pixels determine the percentile stretch.
Centering, identity association and scale effects are not separately identified.

## Coverage And Fixed Protocol

123489 original samples retained, with unchanged labels and row ordering:

| Input handling | Rows |
| --- | ---: |
| Detector-centered dynamic recrop | 76568 (62.003903%) |
| Old-input fallback: no usable detection | 19056 |
| Old-input fallback: MMSI absent from current scene | 15183 |
| Old-input fallback: repeated identity unresolved | 12682 |

Identity joins before detection eligibility: same-FID unique MMSI22617;
remapped unique MMSI71574; unique MMSI-plus-location1433; absent15183;
ambiguous12682. Of the recropped rows, 1317 use the location heuristic.
No rows were removed, including unresolved or visually problematic examples.
The original historical clean-cohort filter was not reapplied to new geometry.

Both frozen encoders were recomputed for changed pixels: paired VV/VH R50
2048D and VH RGB e206 SSL R18 512D. Preserve original spatial/SSL float16
rounding. All46921 unchanged-input rows reuse their exact original features.
Preprocessing took302.56s and encoder inference83.2s, without source refitting.

Reuse `../f0921_transfer_20261007/source_model.pkl`: same source PCA,
29717-row source cohort, source heads and OOF gate. Same target support seeds0..4,
64 labels TOTAL per draw, residual ridge8, identical query masks and 5% caps.
Each draw evaluates123425 rows and admits6171. All five policies were saved
before query readout. `matched_protocol_check.json` records these comparisons.

Five focused unit tests passed (size, borders, old stretch, SSL resize and
identity remapping). Exact old/new feature equality was verified for all46921
fallback rows. The eight crops requesting sides greater than256 were inspected
against raster dimensions; none yielded rectangular truncation.

## Interpretation And Artifacts

This is substantive evidence that the old target input pipeline contributed
to poor F0921 transfer. It does not establish that preprocessing explains all
remaining errors, that the dataset is fully cleaned, or that 29.63% OA is
sufficient absolute performance. Next priority is unresolved input provenance
and radiometry, rather than attributing the gain to another model module.

All earlier dataset limits remain: historically used development cohort,
F0921-exposed SSL pretraining, row-level support/query split, existing source
identity exclusions, no blind-confirmation claim. The 5% cap concerns expert
handoff, not total feature extraction cost.

Original results remain in `../f0921_transfer_20261007` and were not overwritten.
This directory contains `RECROP_PROTOCOL.json`, `crop_metadata.csv`,
`crop_comparison.png`, `feature_recipe.json`, `policies.npz`, `metrics.csv`,
`summary.json` and `input_alignment_comparison.json`.
Implementation is in the sibling `oof_reliability_check` directory:
`prepare_f0921_recrop.py`, `extract_f0921_recrop.py`,
`evaluate_f0921_recrop.py`, `test_f0921_recrop.py`.
