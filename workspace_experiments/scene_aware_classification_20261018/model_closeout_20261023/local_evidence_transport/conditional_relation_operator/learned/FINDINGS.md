# Learned conditional operator: 15 fits complete

2026-10-24. Only learned/ was written. All 15 fits completed 200 full-utility updates in 141.20 s total main execution; all processes exited successfully. No lambda/grid search, ordinary-anchor fitting, external expansion, raw-image analysis, or Git. Parent owns raw cases and literature. Protocol was written before fitting.

**The differentiable operator learns the utility objective, but this fixed protocol does not improve mean J over clean full. Transfer retains a modest BA advantage (+0.3661 pp), paired with OA -0.8935 pp. Excess loses both OA and BA. Do not promote it as a stronger validated algorithm or discard prior retained modules.**

## Selected results

Three-configuration evaluation mean; percentages for OA/BA, probability losses unscaled. Baseline is clean full containing the prior two modules only.

| Arm | OA % | BA % | Delta OA pp | Delta BA pp | NLL | Brier | Selected steps17/29/43 | Net correct17/29/43 |
|---|---:|---:|---:|---:|---:|---:|---|---|
| full | 58.8316 | 40.1223 | 0 | 0 | 1.348249 | .593532 | - | - |
| independent | 57.7320 | 40.2434 | -1.0997 | +.1211 | 1.392716 | .612880 | 0/0/20 | 0/0/-16 |
| transfer | 57.9381 | 40.4884 | -.8935 | +.3661 | 1.369634 | .606135 | 0/0/20 | 0/0/-13 |
| excess | 57.5945 | 39.9534 | -1.2371 | -.1689 | 1.373419 | .606674 | 0/0/20 | 0/0/-18 |
| excess_shuffle101 | 58.8316 | 40.1223 | 0 | 0 | 1.348249 | .593532 | 0/0/0 | 0/0/0 |
| excess_shuffle307 | 58.8316 | 40.1223 | 0 | 0 | 1.348249 | .593532 | 0/0/0 | 0/0/0 |

The small BA-positive scope comes entirely from configuration43; the other two configurations retain identity. Independent also has a smaller BA-positive scope. Excess does not establish an advantage from the analytic permutation correction. Both shuffled arms choose zero; that alone does not demonstrate that real pairing improves the complete model.

For config43, independent rescues18/harms34, transfer20/33, excess18/36. Net changes by class ID0..7:

| Arm | bulk0 | fishing1 | cargo2 | chemical3 | class4 | class5 | tug6 | class7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| independent | -18 | -6 | -1 | +3 | +4 | 0 | +2 | 0 |
| transfer | -15 | -1 | -8 | +8 | +1 | +2 | 0 | 0 |
| excess | -19 | -6 | -6 | +11 | +2 | 0 | 0 | 0 |

Evaluation class counts are 169/91/69/67/33/13/40/3. These counts explain how an OA loss can coexist with a BA gain; they do not prove which image evidence caused a decision change. No majority/minority reinterpretation changes the recorded negative net outcomes.

## What was learned, and what failed to transfer

Initialization and gradients are correct; both body and neighborhood transforms receive nonzero data gradients in all five arms. For real excess, the three-config mean train jointCE falls 1.691746 -> 1.476047 -> 1.056565 -> .699861 -> .521509 at steps0/20/50/100/200. Thus the negative result is not an inactive module or a failure to optimize the requested objective.

At step20, excess selection jointCE slightly improves 1.888448 -> 1.876778 while selection J declines .453948 -> .445447 on average. This is an early probability-risk versus discrete-decision mismatch. By step200 selection jointCE worsens to 4.532710 and selection J to .333336 despite continued utility improvement: a clear training/selection generalization gap under this protocol. Independent, transfer and shuffles also show late utility improvement with selection deterioration. Selected config43 excess improves selection J from .434904 to .458416 but its evaluation J falls from .483822 to .462732; source selection does not ensure evaluation benefit.

Transform Frobenius changes are nontrivial at the selected config43 step20: independent Mb/Mn=2.1258/2.1440, transfer1.9130/1.8588, excess2.0394/1.8915. Excess mean changes grow to9.0178/8.9583 by step200. This supports learned adaptation and later overfit, not a proven claim that transform size alone causes the failure. Final coefficients and all intermediate states remain available for parent tracing.

Data context, reused without a new full audit: utility644 covers587 ships/45 passes/16 ports, selection612 covers562 ships/28 passes/16 ports. Class counts utility=204/83/72/157/26/35/40/27; selection=176/24/51/198/58/36/32/37. Equal macro CE weights correct class frequency within utility; they do not make the per-class observation distributions identical across roles. The bank remains64 frozen references. Prior diagnosis found low-rank within-image predictability but cross-class relation matches even in the high-skill case. The present result is consistent with source-specific fitting that does not deliver reliable class evidence outside utility. The relative roles of observation composition, optimization scale, and transferable representation remain unresolved; no new search was performed to assign a unique cause.

## Implementation and verification

- Shared Mb=I+Ub Vb.T and Mn=I+Un Vn.T, 512 transform parameters plus4 coefficients; all classes/groups share the transforms. Center first, transform second, no renormalization. Exact transfer and permutation-risk formulas retained.
- Independent averages body-to-body and neighbor-to-neighbor reconstruction, activating both maps. This is intentionally different from the parent's neighbor-only fixed independent control.
- Relative update sum_g a_g*(Ztheta-Zidentity), a starts .25, initial prediction exactly full. Utility-only identity RMS per arm is frozen. 12 selected identity states have exactly zero cached relative features, including evaluation rows.
- Six batch128-or-smaller gradients accumulate using complete-utility class counts before one Adam update. Regularizer is applied once per update. The accumulated weighted loss equals dep.joint_loss exactly in the float64 actual-cohort check. Selection checks only steps0/20/50/100/200; strict existing J improvement. Evaluation occurs only after selecting a state.
- GPU vectorization covers all4 groups and64 references. Small Gram products give the same reconstruction scores as explicit neighborhood prediction. No per-batch CPU scalar extraction.
- Parent identity score discrepancy: CPU actual/partial masks2.08e-17, GPU8.60e-9. Scalar null error4.00e-15. Both transform finite-difference gradient errors below1.84e-13. Independent matched-map score error1.39e-17. Initial delta exactly0.
- Selected nonidentity config43 independent/excess traces checked on utility row0 and actual partial-mask row2352: saved GPU versus CPU raw-score error2.51e-8; unpadded scalar versus vector error2.78e-17. Feature-to-logit replay maximum8.94e-7. See smoke.json and trace_verification.json.

## Parent handoff

`METHOD_TRACE.md` documents original-table indexing, selected/checkpoint weight keys, feature meanings, and `trace_query` axes. `weights.npz` includes Mb/Mn, factors and a; `features_*.npz` and `predictions_*.npz` contain selected outputs; all checkpoints are local. `summary.csv`, `metrics.csv`, `class_results.csv`, `paired_results.csv`, `training_curve.csv`, `coefficients.csv`, `scales.csv`, `training_receipts.csv`, and `receipt.json` preserve positive and negative scope. NPZ caches are local outputs, not for Git.

Parent raw review should use config43 for actual learned decision changes and the cached clean-full comparison. Config17/29 and shuffled arms retain identity exactly; their training trajectories remain diagnostic, not selected models. Raw images were deliberately not inspected in this assigned subtask. No strong novelty or physical-mechanism claim is made.
