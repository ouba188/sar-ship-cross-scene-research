# Matched budgeted handoff with a residual expert

Fixed 2026-10-07 before candidate budgeted readout. This follows the positive
full-query residual expert screen; it is not independent confirmation.

## Question and controls

Does the new expert help on the SAME admitted rows under a limited handoff
budget? Keep the original source base as fallback and compare four experts:
fusion, fusion_observed64, anchor_bias64, anchor_visual64. All predictions are
reused unchanged from the verified residual v2 run and coverage control.
No fits, new labels, parameter sweep, knowledge features or new neural modules.

23 active ports, five original uniform support draws, exactly64 labels/draw.
The deployment/evaluation batch is query minus these64 rows. Budget B is
an admission CAP floor(B*n_evaluation), B=.05/.10/.20. No minimum-one or forced
extra admissions: cap0 means select none. Primary B=.05; .10/.20 are secondary
trend points, not a search for the best reporting budget. Budgets are nested
alternatives, not additive expenditures. Target isolation remains row-only.

Two fixed selectors, neither uses query truth or chooses based on new expert
outcomes. Within each selector and support draw, all four experts receive
IDENTICAL masks. Stable ties use ascending global query row id.

- old_lr_cap (primary): original archived source-only four-confidence OOF
  LR0 scores, restricted to original base/fusion disagreement, support-observed
  original expert class and score>0. Rank descending LRscore, take up to cap.
  If candidates are insufficient, keep the budget unspent. Do not refill
  based on a new expert's prediction or drop admissions that do not change base.
- base_uncertainty (simple secondary reference): rank ALL deployment rows by
  ascending top-two original BASE score gap. Take exactly cap. No query expert
  output is needed to select these rows. Not proposed as a new acquisition rule.

The first selector is the old UNCAPPED OOF gate with a declared admission cap,
not a bitwise reproduction of historical s2's5% policy. The earlier s2 script
rounded a budget on a batch including support; this continuation floors a cap
on the unlabeled deployment batch. Do not compare historical aggregate scores
as if their evaluation/masks were identical.

## Costs and deployment boundary

Charge each admitted row whether the receiving expert changes the prediction
or not. Also report actual changes separately; these can be less than admissions.
All four experts have the same admitted-row count at a fixed selector/budget.
The64-label adaptation overhead is separately stated and unchanged.

The OOF selector reads original fusion outputs before admission, so5% admitted
does NOT imply5% expert feature computation. The base-only selector does not
read expert query outputs when choosing its mask, but its full representation
cost and actual selective inference latency have NOT been benchmarked.
Cached complete predictions serve offline replay, not timing measurements.
Do not claim95% compute savings or20x speedup from this study.

## Readout and decision

Save every port's masks and resulting predictions BEFORE reading candidate
query outcomes. Equal-port mean BA after five-draw means is primary; report
accuracy, recoveries/harms relative to base, actual admission/change fractions,
per-class recall, paired port-bootstrap descriptive intervals, positive ports
and draw deltas. Retain all ports, including prior Mombasa degradation.

Primary development screen: at5%, old_lr_cap/anchor_visual64 must have positive
mean BA differences against BOTH old_lr_cap/fusion and old_lr_cap/anchor_bias64.
Also report the stronger observed-class control and the secondary selector;
do not change the primary decision after seeing them. Statistical significance
is not required for this development screen; positivity is not novelty proof.

If this screen passes, prioritize the expert+handoff combination for further
work. If it fails, retain the full-query expert finding but do not claim its
benefit transfers to the original budgeted selection. Do not rescue the result
through portwise routing, budget selection, threshold or regularization scans.
