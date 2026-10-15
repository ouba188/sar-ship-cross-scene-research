# Module completion runs, 2026-10-08

User request: run the missing experiments in the module dossier, not only list missing cells.

## Fixed comparison

- Four arms: matched reference, complement only (M2), deployment-aligned tail only (M3), joint M2+M3.
- Same64 target support labels, seeds0--4, fixed adapted base, fixed gate within each matched comparison, at most5% query handoffs. RGB uses a rebuilt reference gate; ports/publicSAR retain the inherited gate.
- Six source-only tail-training epochs, learning rate 1e-5, AdamW weight decay 1e-4, query batch 128, final checkpoint only.
- Preserve target-specific source identity/product exclusions in BOTH neural training and source head fitting.
- Keep complete negative results. Do not change parameters after observing new target outcomes.
- Existing F0921 joint result remains unchanged; do not rerun it without a specific failure to fix.

## Execution

1. COMPLETED: OpenSARShip1/2 target-specific source episodes, both trained tail arms, original-preprocessing target features, and all four-arm evaluations. Exact historical reference replay passed. See SAR_EXTERNAL_FINDINGS_ZH.md.
2. COMPLETED: Self-built24-port corpus, all23 eligible targets and five draws. Source preparation:506 episodes and253 cached symmetric geometries. All46 tail trainings, all target feature extractions, and460 matched evaluation rows are complete. Melbourne has64 rows and no query after support selection. All23 feature sets and four-arm policies were saved before matched query readout. Joint deltaOA/BA:+0.307381/+0.395702pp; BA positive15ports, negative8.
3. PACS AND OFFICE-HOME COMPLETED: Original-feature M2 and new four-arm runs cover all eight outer domains. Historical RGB R18 weights were not saved, so the four-arm continuation uses an explicitly separate retrained_rgb_reference. Both complete four-domain means are reported; both joint BA changes are negative. All generic and outer-port GPU/CPU stages have exited.
4. FMoW/Camelyon17: no current-protocol result may be inferred from old results or feature inventory. These require a separate compatible-encoder/protocol setup if the necessary artifacts are absent.

## Resource and verification scope

- One new GPU stage at a time. Source pair preparation now uses eight CPU threads on the 24-logical-core host; other workers use two. Preserve the unrelated running RF-DETR task and collaboration applications.
- Reuse original cached raw features and source R18 prefix. Do not re-extract unchanged base features.
- Keep the 1.25 GB frozen SAR prefix on GPU for tail episodes; unchanged float16 values avoid random host paging. RGB bootstrap uniformly uses cached resized uint8, batch64, bf16 autocast with float32 model/Adam state, and all12 source epochs. The aborted initial float32 timing run is preserved separately, not scored.
- Before training, check source episode exclusions. Check differentiable solver against the deployed NumPy solver before each trained arm.
- Before accepting a result, replay reference predictions/admissions and validate support exclusion, shared support indices and handoff counts.
- These checks detect actual leakage, numeric geometry mismatch, or a changed comparison; a failure blocks reporting the affected result until fixed.
- Existing projection operator tests passed: 12 tests. Additional local integration checks are run as new datasets are connected.

Six core dataset groups are complete. See COMPLETION_FINDINGS_ZH.md and CORE_MATCHED_RESULTS.md. Dataset summaries and saved policies identify the completed results; optional FUSAR/FMoW/Camelyon joint extensions remain outside this closeout. No current-run training or evaluation process remains active.
