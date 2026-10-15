# Current-source to F_0921 transfer, fixed 2026-10-07

Use the existing paired chip-level clean rows (165880 of385650) as the
pre-existing target scope, not the unrelated6000-row scene-level all caches.
Map explicit compatible fine labels from class_raw to the original source
eight IDs. Do not reuse the historical M8/class8 broad mapping that turns
generic cargo, generic tanker or LNG/LPG tanker into unsupported fine classes.
Do not introduce new image-quality, per-class-size, or target-outcome filtering.

Freeze the same three policies: original source base, globally adapted base64,
same64 adapted branches with at most5%expert handoff. The64annotations are
TOTAL per target dataset per draw, not64per port. Uniform no-replacement
seeds0..4, no class quotas. Evaluate every non-support row of the compatible
cohort; save all five policies before query-label readout.

Source PCA128/PCA64 and balanced ridge(alpha1), source-port OOF four-feature
logistic gate(C1), source-normalized residual ridge(strength8) in both branches,
observed-expert-class guard, gate score>0, cap=floor(0.05*N_eval), manifest-row
tie break. Source identities are purged against all pre-mapping clean rows.
Use actual Sentinel product identities/candidates, not equality between
F0921 date/UTM tile names and Sentinel scene IDs. Record the actual mapping
scope and any unresolved provenance. Original source inner folds still purge
held-out source MMSIs/products. Do not reuse another target's OOF cache.

The source base is true paired VV/VH through a frozen TWO-channel R50 V2;
separate per-polarization R50 embeddings cannot substitute for it. Extract
paired features from the provided uint8 chip memmaps using the existing
PIL-bilinear128, (x/255-.5)/.25 and bfloat16/spatial-float16 pooling recipe.
The VH SSL cache uses the same e206 R18/Resize128/ToTensor recipe. Reuse it
only after row alignment and a fixed8-row encoder replay (relative L2<=0.001,
allowing its float16 storage). No encoder training or target tuning.

Batch chip reads and target projection keep memory bounded; preserve complete
row IDs and checkpointed extraction progress. Do not subsample the target
for speed or choose the reported subset after observing results.

Report BA, accuracy, per-class recalls, admission counts and all five paired
handoff increments. Five overlapping draws are not independent datasets.
Target support/query stays row-level, not vessel-disjoint. F0921 has historical
development usage AND the e206 encoder was pretrained on F0921. This is a
known-corpus/transductive-encoder transfer experiment, not new independent
external confirmation. The5%cap is expert admission, not encoder-inference cost.

Existing F0921, OSS2, FUSAR and main manuscript results remain untouched.
