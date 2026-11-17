# Conditional modulation summary

Split: evaluation; candidate: affine_frozen; endpoint: validation-selected best.

| Arm | Seeds | OA mean | BA mean | NLL mean |
|---|---|---:|---:|---:|
| affine_frozen | [17, 29, 43] | 64.638 | 59.332 | 1.113527 |
| head_only | [17, 29, 43] | 64.903 | 59.522 | 1.113829 |
| initial_full | [17, 29, 43] | 64.727 | 59.396 | 1.115540 |
| old_broadcast | [17, 29, 43] | 64.638 | 59.332 | 1.113951 |

Missing predictions (not replaced by another arm): {"17": ["broadcast_head", "affine_head"], "29": ["broadcast_head", "affine_head"], "43": ["broadcast_head", "affine_head"]}
Physical completed fits: 14; see fits CSV for trainable counts and budgets.
Old broadcast uses its frozen best, including when current arms are last-weight diagnostics.
Repeated seeds do not multiply independent observations. Evaluation has historical exposure.
Gallery, when requested, describes image evidence; physical mechanisms are not established by writeback maps.
