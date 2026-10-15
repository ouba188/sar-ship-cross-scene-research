# Class-conditioned OOF reliability: fixed development experiment

Date: 2026-10-07. Written before any new candidate result.

## Question and boundary

Test whether the earlier four confidence features discarded useful class
information, then whether a small nonlinear model helps with identical inputs.
This is stronger-baseline development, not a new invention or blind validation.
Reuse the existing 23 true nested-head OOF archives; no encoder, PCA or Ridge
classification-head training. Melbourne remains excluded as an outer query.

Keep the prior five support seeds, 64 distinct target rows per seed, and
query-minus-support evaluation exactly unchanged. Support/query isolation is
row-level, NOT vessel/product-disjoint. The head-training outer/inner identity
purges remain inherited. The encoder-independent status is a user declaration.

This run is free model selection, matching the latest Hermes experiment:
there is NO 5% cap. Report replacement counts. Do not mix these results with
the older budgeted E/G results or claim saved expert computation.

## Fixed models

- `conf_ridge`: previous 4D margins/entropies, same weighted mean squared
  relative-utility loss and lambda=0.01, including intercept penalization.
- `rich_ridge`: 4D confidence + two 7D Helmert logit contrasts + 64 one-hot
  ordered (base, expert) prediction pairs. Total 82 input coordinates.
  Standardize only the 18 continuous coordinates using source-port/class
  weights; leave pair indicators 0/1. Same ridge loss/lambda as conf_ridge.
- `rich_hgb`: exactly the same 82 coordinates, squared-error histogram
  gradient boosting, 100 iterations, learning rate 0.05, at most 7 leaves,
  min_samples_leaf=30, L2=1, max_bins=63, random_state=0, early_stopping=False.
  Source disagreement weights are rescaled to sum to disagreement count;
  this is a fixed ordinary nonlinear baseline, not a loss-matched optimizer.

Source weights: equal port then equal present class within port, computed
BEFORE filtering disagreement rows. Reward = I(expert=y)-I(base=y), including
zero for both wrong. All source models fit only genuine OOF source predictions.

Each model has `0` and `64` versions. `0` means no target coefficient update,
NOT zero target-label use: all arms retain observed[expert] from the same 64
labels. `64` fits a class-balanced linear residual on the same standardized
input plus intercept, with fixed ridge strength 8. For a ridge source model,
this is algebraically the existing source-anchored update. For HGB it corrects
the fixed source score; it does not refit trees on 64 labels. No model selection
between 0/64 per port or after seeing target outcomes.

## Comparisons and outputs

Primary feature contrast: rich_ridge64 - conf_ridge64.
Primary expressiveness contrast: rich_hgb64 - rich_ridge64.
Report both 0-version counterparts, both new models versus simple and the
previous OOF LR, and each adaptation-minus-no-adaptation contrast. Preserve
all signs; do not select a new primary result post hoc. Positive average BA
is a research-retention signal, not proof of novelty or universal superiority.

All new scores/predictions are saved for all ports before new query-label
readout. Store support rows, query rows, eval masks, scores, predictions,
coefficients and source model parameters. Replay simple/s2 and old utility
controls; compare them against prior per-port/seed metrics. Aggregation is
equal port after mean over five overlapping support draws. Bootstrap intervals
and Wilcoxon are descriptive development statistics, not independent trials.

Source and query labels are already historically exposed. This order prevents
new outcome-based switching; it does not erase prior exposure. No parameter
grids, new seeds, global permutations or new target data in this run.

## Tests that can change execution

- Known confidence-feature symmetry versus class-sensitive richer features.
- Source ridge against an independent sklearn weighted solve.
- Residual adaptation against the existing anchored 4D implementation.
- Zero-disagreement identity and observed-expert guard.
- Saved-policy BA from an independent confusion-matrix replay; abort a report
  on archive/control mismatch, never edit a number to pass.

Parallel facility-coverage audit does not alter this protocol. Any knowledge
branch requires a separately recorded fixed design and reliable chip joins.
