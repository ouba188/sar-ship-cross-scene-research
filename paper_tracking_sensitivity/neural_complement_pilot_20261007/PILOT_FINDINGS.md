# F0921 matched representation-learning pilot, 2026-10-07

## Decision

The utility-tail candidate beats ordinary supervised fine-tuning in OA and BA
on all five fixed support draws. It does NOT dominate the existing frozen
expert: mean BA improves 0.213044 percentage points, while OA falls 0.068382.
Keep this as a promising training signal, not a replacement main model or a
validated novel attention mechanism. The existing main results are untouched.

User adoption update, 2026-10-08: retain this as a usable BA-oriented development
candidate despite its small OA decrease. Both metrics need not improve for a
variant to be useful. See ../CANDIDATE_MODULES.md. The training/deployment head
mismatch is being addressed separately; the measured positive result remains.

## What Actually Ran

- Corrected F0921 inputs, 123489 target rows; exactly the previous five global
  64-label supports and 123425 query rows per draw. Cap: 6171 handoffs per draw.
- Original paired VV/VH R50 base features and adapted-base predictions remain
  unchanged. All three arms passed exact base/support/query/budget agreement.
- Expert branch: e206 VH RGB R18. Freeze prefix through layer3; train layer4
  and a training-only eight-class head. Freeze all BatchNorm running statistics.
- Train only on the existing identity-purged 29717 source rows. Fixed six
  epochs, batch128, AdamW lr1e-4/weight decay1e-4, gradient norm cap5,
  class-balanced CE, seed20261007, no augmentation or checkpoint search.
- Source-only 64-label episodes supply adapted-base teacher logits. Their
  28402 query rows receive utility supervision; episode supports and tiny
  source ports receive ordinary classification supervision only.
- Candidate adds a three-outcome auxiliary head: recovery, harm, neutral
  versus the source episodic base. Inputs are 512 visual features, eight
  base logits and eight detached neural expert logits, with a 64-unit hidden
  layer. Loss is CE + utility CE + CE on top predicted-utility 5% of eligible
  minibatch rows; all coefficients are one, fixed before readout.
- Minibatch selection uses ceil(5%) and at least one row. It is a training
  proxy, not the exact deployment selection rule or a budget guarantee.
- Deployment discards the auxiliary/neural classifier heads. Refit the same
  source PCA64, joint ridge expert, head-OOF gate, and target ridge8 residual.
- Frozen replay shares the new float16 prefix cache and float32 tail inference
  with both trained arms. Both trained arms share initialization and batch order.
  All three arms and all five policies were frozen before any target readout.

## Results

Percentages, averaged over five overlapping support draws, not five independent
datasets or five neural training seeds. BA remains the existing primary metric;
OA is reported alongside it, including the unfavorable contrast.

| Representation | Base64 OA | Base64 BA | Full5 OA | Full5 BA | Handoff BA gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen replay | 28.937087 | 24.901788 | 29.634353 | 25.106684 | +0.204897 |
| Ordinary supervised layer4 | 28.937087 | 24.901788 | 29.247559 | 24.997577 | +0.095789 |
| Utility-tail layer4 | 28.937087 | 24.901788 | 29.565971 | 25.319728 | +0.417940 |

| Candidate minus control | OA pp | BA pp | OA positive draws | BA positive draws |
| --- | ---: | ---: | ---: | ---: |
| Supervised minus frozen | -0.386794 | -0.109108 | 0/5 | 2/5 |
| Utility-tail minus supervised | +0.318412 | +0.322151 | 5/5 | 5/5 |
| Utility-tail minus frozen | -0.068382 | +0.213044 | 2/5 | 4/5 |

The earlier corrected-input run was OA29.633543/BA25.105149. Its tiny difference
from frozen replay is not attributed to learning; the recached arithmetic
control is the comparator for all training contrasts above.

## What The Fixed Predictions Explain

The full-expert columns below apply the expert to every query row, solely as
a diagnostic. They violate the 5% deployment cap and are not proposed results.
Counts are means across the five draws, so fractional values are intentional.

| Representation | Full-expert OA | Full-expert BA | Selected recoveries | Selected harms | Recovery BA pp | Harm BA pp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen replay | 31.055945 | 25.982723 | 1798.2 | 937.6 | 1.320424 | 1.115527 |
| Supervised | 29.613449 | 24.485348 | 1701.4 | 1318.2 | 1.011638 | 0.915849 |
| Utility-tail | 30.520235 | 25.420260 | 1878.6 | 1102.4 | 1.235297 | 0.817357 |

Relative to frozen replay, candidate weighted recovery decreases by 0.085127pp
but weighted harm decreases by 0.298171pp, producing the 0.213044pp BA gain.
Unweighted harms increase, explaining why OA does not improve. The candidate
is not an across-the-board better expert. Ordinary source supervision alone
also does not improve transfer under this specific recipe.

The useful next research hypothesis is to align learning with the deployed
64-support residual expert. Here utility labels describe a neural classifier
that is discarded, while deployment uses PCA/ridge plus target adaptation.
That mismatch is structural, but its responsibility for the performance loss
has NOT been experimentally established. A support-conditioned training
objective is a more direct next intervention than an unmotivated attention
addition. Do not expand a hyperparameter sweep on these target outcomes.

## Scope And Verification

- This is a six-epoch layer4 pilot, not full encoder optimization and not an
  implementation or validation of class-pair spatial attention.
- Source head-OOF folds exclude held-port identities, but the supervised
  encoder saw source inner-held labels. This is conditional head-OOF, not
  encoder-level OOF. It limits interpretation of source gate fitting.
- e206 already used F0921 unlabeled pretraining data. The target is historical
  development data with row-level support/query, not a blind confirmation set.
- No target labels fine-tune the encoder. Existing 64 labels adapt both heads.
  No additional annotation budget was spent. The 5% cap is handoff count,
  not feature-inference cost. Training epochs/batches match, not exact FLOPs.
- Three focused loss tests passed: ordinary weighted-CE equivalence, utility
  outcome/mask gradients, and zero-eligible fallback. Training and downstream
  evaluation exited successfully. Selected recovery-minus-harm counts match
  measured OA changes in every arm/draw.

## Reproduction

Scripts are in `E:/Hermes/paper_tracking/sensitivity/oof_reliability_check`:

1. `prepare_neural_complement.py --out <new-output>` with ClearSAR Python.
2. `train_neural_complement.py --out <new-output>` with the existing CUDA env.
3. `evaluate_neural_complement.py --out <new-output>` with ClearSAR Python.
4. `read_neural_complement.py --out <new-output>` for fixed-prediction diagnostics.

Read `protocol.json`, `TRAINING_COMPLETE.json`, `summary.json`, and
`diagnostics.json` here. Per-arm checkpoints, training histories, source OOF,
source model, policies, and metrics are retained. Prefix caches are shared
within this run; do not reuse with changed input geometry or manifest order.
