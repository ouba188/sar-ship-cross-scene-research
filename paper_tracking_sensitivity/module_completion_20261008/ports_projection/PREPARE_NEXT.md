# Outer-Port M3/Joint Preparation

The CPU preparation interface is implemented in `run_ports_projection_completion.py`.
It must run only after `COMPLETE.json` from the matched M2 evaluation exists.
Full 23-port preparation was launched on 2026-10-08 after the matched M2 process completed.
Each per-port `EPISODES_COMPLETE.json` is the readiness marker for the independently scheduled GPU stage.

```powershell
& 'D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe' `
  'E:/Hermes/paper_tracking/sensitivity/oof_reliability_check/run_ports_projection_completion.py' `
  --stage prepare --threads 8
```

Use `--ports 'Antwerp-Bruges'` to prepare one outer target for a queued GPU smoke run.
Omitting `--ports` prepares all 23 eligible outer targets. Completed episodes are reused.

## Per-Outer Outputs

- `source_model.pkl`: independently reconstructed PCA and source heads for this outer split; inherited source OOF gate.
- `source_geometry.npz`: source-only normalization and the exact outer source/query row lists (already produced by M2).
- `source_rows.npz`: `source_rows`, full-manifest `labels`, and `target_rows`; supervised training accesses only source rows.
- `deployment_complement.npz`: pretraining source-only complement center/scale; frozen after training.
- `protocol.json`: compatible with `sar_joint_completion.train_arms(out)`; training constants, original e206 initialization, prefix and row paths.
- `episodes/*.npz`: `rows`, `train_rows`, local `support/query`, `labels`, retained R50 coordinates, complement R50 coordinates, and fixed SSL/head geometry.
- `EPISODES_COMPLETE.json`: list of episodes plus actual query counts and outer-exclusion declarations.

Every inner prior fit excludes both outer-target identities and the held source-port identities.
Ports with at most 64 held rows have no nonempty query episode and are skipped as in the existing helper.
The outer evaluation itself retains all 23 original eligible targets and all five 64-label draws.

## Cache and Resource Policy

`episode_geometry_cache/` stores each unordered pair of excluded ports once.
Reuse requires exact ordered `train_rows` equality, not just equal port names.
This replaces repeated PCA/source-statistic fits; it does not change folds or training budget.
CPU preparation memory-maps raw R50 features and does not load the frozen layer3 tensors.
The current GPU helper preloads the 1.25GB float16 layer3 array to GPU in chunks,
then indexes the identical values by episode rows. This changes storage location, not the experiment.
Only the existing 512-D SSL feature array is resident (about 78MB).
CPU preparation started at two threads, resumed at four after the first complete outer port,
and then resumed at eight with the completed outer-port and pair caches retained.
The cached-geometry test compares against the actual generic SAR helper.

## GPU and Evaluation Handoff

After CPU preparation and a GPU slot are available, `--stage train --ports '<port>'`
calls `sar_joint_completion.train_arms(port_directory)` and extracts the three target feature sets.
It trains the two six-epoch arms independently from the original e206 layer4, never from an F0921-trained checkpoint.
Use `neural_complement_pilot_20261007/source_layer3.npy` indexed by this outer target's `target_rows`
to extract only the held target rows through each trained layer4. Do not extract all 38091 rows per outer model.

For deployment, use the exact cached `pred_cache2/<port>.npz` R50 `Xs[target_rows]`.
Only the SSL encoder features change. Transform those through the frozen per-outer SSL PCA/head geometry,
then apply the actual ridge8 shared64 solver and original Base64/gate/5% policies.
Original Base64 policies are at `oof_base_adapt_results_20261007/<port>/policies.npz`;
the new per-port protocol records that absolute path.

The old R50 PCA coordinate cache is not bitwise identical to current PCA refits.
M2 keeps its retained coordinates and prior scores exactly unchanged; it uses reconstructed PCA only for complement geometry.
Per-port `POLICIES_SAVED.json` reports the numerical reconstruction discrepancy and exact reference-policy replay.
The initial frozen-prefix R18 replay is separately extracted and reported before interpreting trained-tail effects,
because its layer3 cache is float16 whereas the old SSL feature archive is float32 inference.
Do not silently attribute a changed preprocessing or source geometry to M3 gains.

No GPU work is performed by `--stage prepare`.

## Scheduled GPU Command

Run only after GPU scheduling approval. Change the port name to each ready outer fold.

```powershell
& 'E:/Hermes/paper_tracking/envs/msfa-openmmlab-py310-rtx50-cu132/Scripts/python.exe' `
  'E:/Hermes/paper_tracking/sensitivity/oof_reliability_check/run_ports_projection_completion.py' `
  --stage train --ports 'Antwerp-Bruges'
```

The stage does not read target query outcomes. It writes each port's
`matched_joint/FEATURES_COMPLETE.json` and `reference_extra.npy`, `aligned_only_extra.npy`, `joint_extra.npy`.
The new initial-tail reference uses exactly the same float16 prefix cache as the trained tails.

## Complete Matched Readout

After all23 outer checkpoints and extracted feature sets exist:

```powershell
& 'D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe' `
  'E:/Hermes/paper_tracking/sensitivity/oof_reliability_check/run_ports_projection_completion.py' `
  --stage evaluate
```

All23 ports' four-arm policies are saved before query readout. Results are written to
each port's `matched_joint/metrics.json`, `metrics.csv`, and `summary.json`.
Each summary includes `new_reference_minus_old_reference_pp`; old cached-reference results are preserved.
The complete equal-port aggregate is `ports_projection/joint_summary.json`, with
`matched_joint_metrics.csv` containing all460 port/draw/arm records.
The historical M2 `ports_projection/summary.json` is never overwritten by train/evaluate.
