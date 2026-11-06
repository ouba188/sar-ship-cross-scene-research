# M1 class-competitive covariance pilot

Status: implementation and bounded development experiment; no formal model change or Git sync.

Completion note: all12 initial fits plus6 conditional frozen-choice repeats completed; see FINDINGS_ZH.md. The original protocol below is retained. A post-fit covariance-zero diagnostic was added without changing trained forward paths or checkpoint choices to separate covariance values from response-only/bias effects.

## Question and data evidence

Prior high/low M1 and separated-deviation repair tied the strongest tuned mean on all 228 OSS2 predictions. Removing the deviation branch caused no top-1 change. Query003648 has compact bright patterns also present in general-cargo references; query003630 has elongated patterns also present in bulk references. These facts motivate testing relationships, not claiming a physical mechanism or guaranteed gain.

Use the existing OSS2 four-class source719 / historical development228 / fixed32 source-reference cohort. No selection194 or evaluation378 access. Native160 crop/pad, VV/VH separate, source/query vessel and pass separation unchanged. Missing polarization is masked. This is exploratory development selection, not independent validation.

## Operators

- mean: weighted spatial mean, common and parallel nonlinear mean transformations.
- second: rank4 x width32 centered response-feature bilinear context, without direct class alignment of the four response probes. This is a matched low-rank second-order comparator, NOT a reproduction of full GSoP-Net.
- class: centered one-versus-rest responses, class-indexed covariance tokens and transformations; direct class-response auxiliary logits plus a joint mean/relation auxiliary head.

All three have identical tensor widths and parameter allocations. Same-seed common initialization; all use mean path + bounded residual multiplication, zero output initialization. Mean is a strong parameter-matched control, and prior best mean is additionally retained. Auxiliaries differ intentionally in semantic alignment; success would support the package, not independently identify each design contribution. Learned feature responses are not physical scattering labels.

## Placement and adaptation

- late: layer3[5] after conv2/BN/ReLU, before conv3.
- early: layer2[3] at the analogous location, before conv3.

Only one M1, not both. Old M1 disabled; existing M2 stays at layer3[5] before conv2. Same E17 public carrier (without private layer2[2] M2). Train inserted M1, that block's conv3, and classification head; other parameters/BN fixed, but gradient flows through fixed later layers. Early insertion recomputes downstream feature and M2 shallow summary. M3 protected source-reference message stays the same frozen cached path, explicitly a bounded integration pilot rather than full reference relearning.

## Bounded selection

Initial seed17: 3 operators x 2 positions x gains0.1/0.3, each six epochs/180 steps, identical source orders, batch24, source class/vessel weighting and prior hard weighting. Internal LR3e-4, output LR1e-3, adjacent conv3/head LR1e-5, auxiliary coefficient0.02, AdamW wd0.01, gradient clipping5. Select checkpoints by development BA/OA/NLL/earlier epoch including epoch0. No unlimited sweep. Compare each method's best independently, also matched settings and fixed last epoch.

If the class candidate beats independently selected mean and second-order on development BA with nonnegative OA change, freeze its position/gain/epoch and repeat all three at seeds29/43 using each method's frozen seed17 epoch. Otherwise inspect gradients, relation-off/class-merge diagnostics and native chips, and stop this round with the actual result. Preserve all failed trials.

Checks target concrete failures: zero-start replay vs no-M1 carrier; live shallow-path parity against stored layer2 features; finite/nonzero first-step output and later relation gradients; constant spatial field gives zero covariance; invalid padding does not affect context; restored selected checkpoint reproduces predictions. No changed-label or oracle selection.

After fitting review rescue, harm, stable-correct and stable-wrong native VV/VH chips and actual legal source references. Relation-off is same-weight diagnostic, not a substitute for matched training. Report class composition, continuous margins, and runtime alongside OA/BA.

## Prior-method basis

Bilinear CNN (ICCV2015) motivates local co-occurrence descriptors; GSoP-Net (CVPR2019) motivates using second-order statistics for intermediate feature modulation and testing placement. Neither validates this class-competitive implementation or our cross-acquisition setting. Existing mean/high-low and position experiments remain the direct task-specific basis.
