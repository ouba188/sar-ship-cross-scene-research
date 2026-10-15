# Stable-guided spatial pooling pilot

Completed six fixed ports, five existing64 draws each, CPU ClearSAR with two threads. All six policies froze before target query evaluation. Historical main model and base scores are unchanged.

| Arm | OA % | BA % | Expert BA % | Rescues | Harms | Admissions |
|---|---:|---:|---:|---:|---:|---:|
| original | 44.5655 | 45.0082 | 48.6416 | 63.03 | 23.23 | 138.83 |
| matched_refit | 44.4739 | 45.0519 | 48.6416 | 61.73 | 25.20 | 138.83 |
| ordinary_pool | 44.0577 | 44.7399 | 43.5238 | 54.80 | 25.03 | 138.83 |
| stable_pool | 44.1415 | 44.7623 | 43.6568 | 54.97 | 24.77 | 138.83 |

Metrics are equal-port means after averaging five draws. Counts are mean counts per port/draw.

## Matched comparisons

- matched_refit minus original: OA -0.0915 pp; BA +0.0437 pp; expert BA +0.0000 pp.
- ordinary_pool minus matched_refit: OA -0.4162 pp; BA -0.3120 pp; expert BA -5.1178 pp.
- stable_pool minus matched_refit: OA -0.3324 pp; BA -0.2896 pp; expert BA -4.9848 pp.
- stable_pool minus ordinary_pool: OA +0.0837 pp; BA +0.0224 pp; expert BA +0.1330 pp.

| Port | Ordinary vs matched BA pp | Stable vs matched BA pp | Stable vs ordinary BA pp |
|---|---:|---:|---:|
| Antwerp-Bruges | -0.2880 | -0.0432 | +0.2447 |
| Jebel Ali | -0.3175 | -0.2639 | +0.0536 |
| Port Klang | -0.3553 | -0.1858 | +0.1695 |
| Santos | -0.0703 | -0.2978 | -0.2275 |
| Singapore | -0.2447 | -0.2358 | +0.0089 |
| Tanger Med | -0.5960 | -0.7111 | -0.1151 |

## Pipeline and scope

Source-base predicted top-two class pairs select ordinary pooled or conditional-penalty directions. Absolute within-chip centered token evidence, divided by per-direction source RMS, enters temperature-one softmax. The mean-zero attention residual is projected into the original 128 PCA coordinates and scaled by source RMS. A balanced ridge1 residual head fits desired scores minus the historical source expert prior. The target64 ridge8 correction is refit jointly with that augmented prior and the original expert kernel plus .25 times the normalized spatial kernel. The base is byte-for-byte unchanged. Each new arm refits the same four-confidence-feature LR C1 gate; the matched control only refits that gate. The historical original keeps its saved gate. This pilot uses existing binary fit_gate, which drops zero-utility rows and learns rescue versus harm on informative eligible disagreements. The concurrent drift-response pilot retains zero-utility rows in a net-utility classifier, so its refit control is not directly the same baseline.

Residual heads and pair-selector heads exclude held source labels using existing purged train rows. Directions, PCA, RMS moments and encoder use all outer source, so this is NOT fully inner-OOF geometry. Source expert metrics use held episodes; routed source metrics reuse gate fitting episodes and are descriptive, not an independent gate evaluation. Only existing64 target labels enter fitting; all query labels are read after freezing.

Historical token means exactly reproduce the original pooled features on seven fixed alignment rows; new-cache manifest identities also match. Original expert-kernel replay is checked for all source episodes and target draws. The focused tests cover row mapping/mean reconstruction, uniform no-op, label-free predicted-pair inference, source-only scaling, explicit primal/dual consistency and zero-branch replay.

## Minimal diagnosis

- ordinary_pool: source-held expert BA delta -5.3104 pp; mean held prior addition RMS 0.5902; mean changed held expert predictions 300.7.
- stable_pool: source-held expert BA delta -5.1981 pp; mean held prior addition RMS 0.5701; mean changed held expert predictions 293.7.

Nonzero module diagnostics establish that the branch is active; they do not establish transfer value. For this completed pilot, both pooling arms regress on source-held expert BA as well as target expert BA. Thus the negative result is already present before target routing; target-only shift or a no-op does not explain it. The remaining cause within attention representation, residual fitting and regularization is unresolved. This fixed recipe does not support merging the new expert into the drift-response gate. Negative results and every port are retained. No target parameter search or physical interpretation was performed. These are development comparisons, with overlapping draws and no independent confirmation. Ordinary versus stable isolates this guidance choice within the matched pooling implementation; it does not identify stability causally.
