# OpenSARShip1 current-method transfer, 2026-10-07

## Result

All values are percentages, averaged over five fixed support draws.

| Arm | Ordinary accuracy | Balanced accuracy |
| --- | ---: | ---: |
| Original source-only base | 38.163424 | 21.233037 |
| Base adapted using global64 target labels | 40.513619 | 31.510334 |
| Full model with5%expert handoff | 43.066148 | 32.330577 |

Handoff adds **+2.552529ppOA / +0.820242ppBA**, positive in both metrics for
all five draws. BA increments:1.084294,0.428258,0.633087,1.277423,0.678151pp.
Each draw evaluates1285query rows and admits64, below the5%cap.
The base adaptation's+10.277297ppBA is NOT an expert-handoff contribution.
Product/chemical-tanker mean recall falls from7.169710%to6.099578%after
handoff, so positive average BA does not imply every class benefits.

## Native Cohort And Representation

The provided local root contains41product directories, not82:
22GRDH products /3030annotations and19SLC products /2643annotations.
There are5673native annotation rows. This is a local inventory observation,
not a claim about the full published dataset distribution.

Primary evaluation uses GRDH for input-product consistency with the source
pipeline. SLC companion products are not pooled into the target evaluation.
Native Ship.xml fine labels are joined to Patch_Uint8 by scene+Center_x/Center_y,
not by file sorting order or the coarse filename category. No quality,
interference, size, historical clean-manifest or performance filter is used.

The existing strict fine-label bridge retains1349/3030GRD rows:
bulk816, fishing35, general-cargo121, product/chemical116, container218,
crude-oil13, tug18, offshore12. All eight original source output columns remain.
Coarse Cargo/Tanker and unsupported subtypes are not relabeled as fine classes.
All compatible samples have both VV and VH images.

Both feature views were newly extracted using the current source recipe:
paired two-channel R50 V2,2048D and VH-RGB frozen e206 SSL R18,512D.
The old opensarship_external/external_feats.npz uses a different RGB base
recipe and is preserved, not substituted. Native8bit chips receive the existing
base bilinear128 normalization; SSL Resize128 preserves aspect ratio.

Source fitting excludes MMSI/product identities from ALL5673native target
rows, including SLC and unmappable labels.748source rows /353MMSIs are
excluded;37343source rows remain. No exact source product IDs overlap.
All24source-port head-OOF folds, PCA, classifiers and source LR gate were refit.

## Fixed Protocol And Limits

Five seeds0..4,64labels TOTAL across this target cohort per draw, ridge1source
heads, ridge8target residual, PCA128/64, fixed LR gate and5%handoff cap.
Support and query are row-disjoint, not vessel-disjoint. All five policies
were saved before query readout. No target hyperparameter selection occurred.

This is a historically used, fine-label-compatible GRD subset, not a new
blind test, not the full standard taxonomy and not a within-OSS1 LOPO result.
It measures24-port-source to OSS1 transfer.5%refers to expert admission,
not feature computation. Native-GRD identity parsing and the existing shared
policy/readout tests passed (five tests in the OSS1 test invocation).

Artifacts: native_label_mapping.csv, label_mapping_summary.json,
source_split.json, target_manifest.csv, features.npz, source_model.pkl,
source_oof/, policies.npz, metrics.csv and summary.json.
Scripts are in ../oof_reliability_check/: opensarship1_data.py,
run_opensarship1_transfer.py, extract_opensarship1_transfer.py.
