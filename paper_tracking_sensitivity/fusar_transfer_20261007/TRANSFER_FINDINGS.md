# Current source -> FUSARShip: completed 2026-10-07

## Result

The fixed handoff has a positive mean increment on this compatible FUSAR
subset, but one of five support draws loses balanced accuracy. Keep that
negative draw; no settings or label rules were changed after readout.

| Fixed policy | Mean balanced accuracy (%) | Mean accuracy (%) |
| --- | ---: | ---: |
| Original source base | 22.463793 | 46.965265 |
| Base adapted with 64 target labels | 28.446428 | 51.608775 |
| Same-64 adapted branches + at most5% expert handoff | 28.894663 | 52.650823 |

- Base adaptation gain: +5.982635 BA percentage points.
- Additional expert-handoff gain: +0.448235 BA points, +1.042048 accuracy points.
- Full method over original base: +6.430870 BA points.
- Handoff increments, seeds0..4: +0.077880, +1.648884, -0.685866,
  +0.651170, +0.549109 BA points. Four positive, one negative.
- Every draw uses the SAME64 target labels for both branches and admits
  54/1094 evaluation rows (4.936015%), within floor(0.05*1094)=54.
- All five policies were saved before query readout. Source models and
  parameters were frozen; no target hyperparameter search was run.

The negative draw (seed2) has BA-weighted recoveries of0.724497 points and
harms of1.410363 points relative to the adapted base. Thus its negative
increment is an actual adverse handoff outcome, not a missing run.

## Coverage and comparison boundary

The existing five-category preparation has3276 images, of which3166 are
eligible after96 shared-main-MMSI and14invalid-MMSI exclusions. Its eligibility
boundary is preserved. Explicit fine-label alignment retains1158 images:

| Source ID | Fine class | Target rows |
| ---: | --- | ---: |
| 0 | bulk_carrier | 247 |
| 1 | fishing_vessel | 731 |
| 2 | general_cargo | 33 |
| 3 | product_chemical_tanker | 28 |
| 4 | container_ship | 57 |
| 5 | crude_oil_tanker | 8 |
| 6 | tug_towing | 54 |
| 7 | offshore_supply | 0 |

Of the3166 eligible rows,2008 cannot be used under this fine-label protocol:
1908coarse labels,83unsupported,8ambiguous,9coarse/fine conflicts. Generic
CargoShip and Tanker were not silently converted to fine classes. There is
no additional quality, image-size, or AIS-dimension filter.

The source still predicts all EIGHT classes; evaluation macro-averages the
SEVEN classes actually present. Offshore recall is null, not zero. This is
a compatible fine-label subset, NOT performance on the whole standard
five-category FUSAR benchmark. The eight crude-oil examples are particularly
sparse; some support draws contain no examples of this class.

Source overlap against all3166 eligible target identities is zero. Therefore
all38091 original source rows remain, and the24source-port OOF folds and
source PCA/heads/gate are refit. FUSAR scene IDs are unavailable, so no scene-
disjointness claim is made. Target support/query is row-level; query rows
sharing a support MMSI are8,11,5,7,10 for seeds0..4, respectively.

## Representation

FUSAR provides one grayscale image in this version. The unchanged two-channel
source R50 receives that SAME image in both input channels, and the unchanged
domain-SSL R18 receives the same image as RGB. This is a zero-parameter input
adapter, not a recovered VV/VH pair and not a second observation. Describe
the run as single-image dual-encoder transfer, not dual-polarization evidence.

The shared extractor matches the preceding OSS2 recipe; its source-cache
replay remains recorded in opensarship2_transfer_20261007/encoder_replay.json.
No backbone is retrained. E206 SSL retains its F0921 pretraining exposure.
The old external_os2_fusar/feats_fusar.npz is preserved but not substituted,
because its base input/normalization and resize recipe differ.

## Interpretation

This supplies another positive mean transfer increment for the current
handoff mechanism, with different input availability. It does not establish
all-draw or all-class improvement. Mean container and tug recalls fall by
about1.07 and1.15 points under handoff, respectively. Absolute BA is28.89%,
so substantial target-domain error remains.

Do not add the base-adaptation gain to the claimed expert-module contribution.
Do not compare this seven-class subset BA directly with the OSS2 eight-class
BA or old within-FUSAR five-class cross-validation results. Five overlapping
support draws are descriptive repetitions, not independent datasets.
FUSAR has historical development usage, so this is not a new blind test.
The5%cap is expert admission, not feature-extraction or inference cost.

## Files and verification

TRANSFER_PROTOCOL.md and RUN_CONFIG.json specify the fixed run. Complete
results are in summary.json, metrics.json/csv, and policies.npz. Mapping and
coverage are in label_mapping.csv, label_mapping_summary.json, target_manifest.csv.
Features and provenance are features.npz and feature_recipe.json. Source fits
are source_split.json/npz, source_oof/, source_oof.npz, source_model.pkl.

Forty focused label-mapping pytest cases passed. Four shared policy/readout
unittest cases passed, including absent-target-class recall handling. All24
source folds and all5target draws completed. No OSS2 result file was changed.

Entry scripts under sensitivity/oof_reliability_check:

- extract_fusar_transfer.py: existing GPU interpreter, --out <new-output>.
- run_fusar_transfer.py: ClearSAR interpreter, --out <same-output>.

Both use existing frozen encoders and helper functions; no dependencies were
installed and no unrelated process was closed.
