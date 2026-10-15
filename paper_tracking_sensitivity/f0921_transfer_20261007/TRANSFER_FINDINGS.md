# Current source -> F_0921: completed 2026-10-07

## Result

The frozen same64/5%handoff method has a positive mean increment on the
existing F0921 clean, fine-label-compatible cohort. Four support draws improve;
one declines slightly. Absolute recognition performance remains low.

| Fixed policy | Mean balanced accuracy (%) | Mean accuracy (%) |
| --- | ---: | ---: |
| Original source base | 18.846647 | 17.920356 |
| Base adapted with64 target labels | 20.441980 | 21.433259 |
| Same64 adapted branches + at most5%expert handoff | 20.839997 | 21.834150 |

- Base adaptation gain: +1.595333 BA percentage points.
- Additional handoff gain: +0.398017 BA points; +0.400891 accuracy points.
- Full method over original base: +1.993350 BA points.
- Handoff increments for seeds0..4: +0.936301, +0.141258, +0.324781,
  -0.004176, +0.591920 BA points. Keep the negative seed3; do not round it
  into a positive draw or report5/5wins.
- Every draw admits6171/123425 evaluation rows (4.999797%). The64annotations
  are TOTAL across this target dataset, shared by both branches, not64per port.
- No target parameter search, label-budget increase or outcome-based cohort
  change was made. All five policies were saved before query readout.

## Target cohort

The input index has385650 chips. Existing paired clean feature archives define
165880 chip rows using the prior land_px_224==0 selection. We inherit that
selection; this is not the raw unfiltered corpus. The6000-row all archives are
scene-level features and are not interchangeable with these chip rows.

Explicit class_raw alignment retains123489 rows across the original eight
classes. Historical class8 is preserved only as metadata, not accepted as
ground truth for this bridge. Old generic cargo/tanker, LNG/LPG and ambiguous
offshore mappings are not promoted to fine classes.

| Class | Retained target rows |
| --- | ---: |
| bulk_carrier | 28880 |
| fishing_vessel | 3800 |
| general_cargo | 7564 |
| product_chemical_tanker | 23229 |
| container_ship | 30638 |
| crude_oil_tanker | 13947 |
| tug_towing | 13357 |
| offshore_supply | 2074 |

Excluded from the existing clean scope:30397unsupported,3605coarse and
8389ambiguous labels. No additional quality, image-size, minimum-class-count
or geographic filtering is introduced. Retained coordinates span longitude
-161.14904..-60.80279 and latitude15.83907..50.10886; none are south of the equator.

## Source identities and fitting

Purge source identities against ALL165880 clean rows, before target fine-label
selection. MMSI overlap affects7899source rows/4442shared IDs. Acquisition
candidate overlap affects1839source rows/146products;1364rows overlap both.
The union removes8374rows, leaving29717source rows with all eight classes.

Target scene names are date/UTM tiles, not Sentinel product IDs. The older
tile_products lookup stores only the first acquisition. Join all recorded
tile acquisition candidates to source products using satellite, start time
and product suffix code. This finds57shared products beyond the first-only
lookup. This is conservative exclusion across tile candidates, NOT exact
chip-level acquisition attribution. Full provenance is in source_split.json.

Source fitting now covers23ports: Los Angeles has no retained rows; Houston
and New York have one each. The unchanged equal-port gate weighting retains
those small source groups, so their individual OOF samples have disproportionate
weight. This is a limitation of applying the frozen weighting recipe after
identity purge, not a reason to silently drop groups or change weights after
target readout. All23OOF folds completed; source PCA, heads and gate were refit.

Target support/query remains row-level. Evaluation rows sharing a support
MMSI number1356,1065,999,1248,1240 for seeds0..4. Thus this is not a target
vessel-disjoint test. Some draws have no fishing/offshore support examples;
the existing observed-expert-class guard is retained without quota resampling.

## Features and encoder exposure

Separate VV/VH R50 caches are NOT substitutes for the current joint two-input-
channel R50. Newly extract2048D paired features from the provided uint8
224x224 chip memmaps with the original bilinear128, normalization and
bfloat16/spatial-float16 pooling recipe. Extraction completed123489rows in
105.3seconds; original caches are untouched.

Reuse the aligned512D VH SSL cache only after8fixed-row replay against e206.
Maximum relative L2 difference is0.000309951, within the predeclared0.001
tolerance allowing existing float16 storage. Store selected SSL as float16
and convert to float32 before source PCA. No encoder retraining is performed.

IMPORTANT: e206 was pretrained on E:/F_0921_pool_v1. F0921 is historically used
development data and was exposed to the SSL encoder. This is known-corpus,
transductive-encoder transfer evidence, NOT independent external confirmation
or an unseen-encoder target dataset.

## Interpretation

The handoff has a small positive average benefit, consistent in sign with the
other two current transfer studies. It does not solve domain shift: full BA
is only20.84%. Nor is it all-class improvement; mean bulk and offshore recalls
decline by about1.01 and0.87points under handoff. Do not attribute base-adaptation
gains to the handoff module, call the small seed3loss a win, or claim statistical
significance from five overlapping draws.

Comparisons are paired WITHIN this run. The much larger target set still gets
only64global labels; its absolute BA is not directly comparable to prior
within-F0921 leave-port experiments or the other transfer datasets. The5%cap
limits expert admissions, not computing all expert features.

## Artifacts and verification

- TRANSFER_PROTOCOL.md, RUN_CONFIG.json: fixed settings.
- summary.json, metrics.json/csv, policies.npz: complete five-draw readout.
- label_mapping.csv, label_mapping_summary.json, target_manifest.csv: cohort.
- source_split.json/npz, source_oof/, source_oof.npz, source_model.pkl: source fits.
- paired_r50.npy, vh_ssl.npy, feature_rows.npy, feature_progress.json,
  feature_recipe.json, ssl_cache_replay.json: row-aligned features/provenance.

Thirty-five cohort/identity tests passed, including actual-data counts. A
focused projection test passed: batching with a partial last batch and float16
SSL inputs matches one-shot projection within float32 tolerance, with unchanged
argmax predictions. Initial1e-6absolute tolerance exposed a2.44e-6BLAS rounding
difference; the documented1e-5tolerance reflects float32 precision, not a change
to the model or outcomes. All23source folds and5target draws completed.

Reproduction order under sensitivity/oof_reliability_check:

1. ClearSAR: run_f0921_transfer.py --out <new-output> --stage prepare
2. ClearSAR: run_f0921_transfer.py --out <same-output> --stage source
3. GPU interpreter: extract_f0921_transfer.py --out <same-output>
4. ClearSAR: run_f0921_transfer.py --out <same-output> --stage finish

Steps2and3can run concurrently. Chips and projection are processed in batches;
the extractor supports aligned-row progress resume. Completed outputs are
protected from accidental overwrite. No dependencies were installed and no
unrelated application was closed.
