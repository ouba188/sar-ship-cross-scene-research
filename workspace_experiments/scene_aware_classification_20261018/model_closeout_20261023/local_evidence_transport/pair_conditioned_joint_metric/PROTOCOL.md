# Pair-conditioned joint-evidence metric

Frozen before fitting, 2026-10-24. Historical development cohort, not independent confirmation.

Motivation: prior common per-image joint-pooling transform rescued30232 but harmed4531 against the same bulk references26615/2379. This motivates, but does not establish, using the particular query/reference relationship to choose content-comparison channels. No assumption that smaller frequency discrepancy means same class.

Keep original images, cached joint C=E[eF]-E[e]E[F] and marginal M=E[e]E[F], source scaler,32dim content x,20 OBB slots, source bank and frozen classifier. For each local token, h=A*d, A:96->4 without bias. Pair feature phi=[(hq+hs)/2,abs(hq-hs)/2]. Channel weight w=1+0.25*tanh(B*GELU(phi)), B:8->32 without bias. A random/B zero. Total640 trainable parameters. Weight range[.75,1.25]. Both query and reference determine each local-edge metric; no class ID or ground-truth label enters conditioning.

K(q,s)=sum(w*xq*xs)/sqrt(sum(w*xq^2)*sum(w*xs^2)). This normalization prevents uniform weight amplification from simply changing the similarity/OT temperature. Symmetric and bounded[-1,1], but not claimed to be a globally PSD kernel or a metric satisfying triangle inequality. Regional OT and oldhead unchanged; final score=strong+head(newlocal)-head(oldlocal). C does NOT directly contribute an independent class score.

Three matched640-param arms:
- pair_joint: joint C and pair phi.
- separable_joint: joint C, weight=1+.25*(tanh(B*GELU([hq,abs(hq)]))+tanh(B*GELU([hs,abs(hs)])))/2. Same parameter matrices, no cross-image interaction in weight generation. Same normalized similarity.
- pair_marginal: M with pair phi. Tests co-location contribution at matched architecture.

Original independent joint-pooling and prior frequency gate remain contextual baselines, not clean single-variable comparisons. New metric normalization is common to all three arms.

Sourcebank64, utility644, selection612, historical development485; source-only scaler/loss/selection. Seeds17/29/43,100 full-source updates,AdamW lr.003 wd.01, half empirical/half equal-class CE, checkpoints0/10/20/40/60/80/100, choose max source J=(OA+BA)/2 with earlier ties. All9 fitted before evaluation metrics. No new image extraction. Identity eligible; keep all results.

Checks target insertion bugs: initial fullscore/local replay, finite-difference live output-gradient, q/ref symmetry, normalized similarity bounds and invariance to uniform positive weight scaling. Pair comparisons and normalization are differentiable; invalid slots zero-mass and old path preserved.

Evaluate full chain OA/BA/NLL/Brier, samequery rescues/harms, classes and acquisition groups. Source-only diagnostic separately examines whether simple joint-distance proximity is reliable. After training inspect native VV/VH query and actual true/rival reference images for rescue/harm/CC/WW; use frozen-head region accounting and compute effect of holding old/new matching T versus metric K when needed. Do not label attention weights as physical cause. No new public dataset or target-label fit.
