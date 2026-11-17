# Conditional modulation summary

Split: validation; candidate: affine_frozen; endpoint: validation-selected best.

| Arm | Seeds | OA mean | BA mean | NLL mean |
|---|---|---:|---:|---:|
| affine_frozen | [17, 29, 43] | 66.082 | 61.318 | 1.165210 |
| affine_head | [17] | 68.860 | 63.042 | 1.163140 |
| broadcast_head | [17] | 68.860 | 63.042 | 1.163116 |
| head_only | [17, 29, 43] | 66.082 | 61.318 | 1.165240 |
| initial_full | [17, 29, 43] | 65.936 | 61.224 | 1.166446 |
| old_broadcast | [17, 29, 43] | 66.082 | 61.318 | 1.165477 |

Missing predictions (not replaced by another arm): {"17": [], "29": ["broadcast_head", "affine_head"], "43": ["broadcast_head", "affine_head"]}
Physical completed fits: 14; see fits CSV for trainable counts and budgets.
Old broadcast uses its frozen best, including when current arms are last-weight diagnostics.
Repeated seeds do not multiply independent observations. Evaluation has historical exposure.
Gallery, when requested, describes image evidence; physical mechanisms are not established by writeback maps.
