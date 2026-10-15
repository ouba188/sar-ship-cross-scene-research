# Fixed adapted-head action-advantage control

Written before this candidate's target readout on 2026-10-07, following the
negative source-episodic gate screen. This is one cheap diagnostic baseline,
not a new attention module or an algorithmic novelty claim. Historical
run_key_improve.py tested an action-pair difference of a different conditional
likelihood posterior; it did not test the present adapted expert under these
budgeted policies.

Use the archived anchor_bias64 and anchor_visual64 scores without refitting.
For each expert, apply softmax with fixed temperature1 to its eight scores.
Rank by p(expert_argmax)-p(base_argmax), with BOTH terms from this same expert.
This compares the two actions being considered rather than two head maxima
or the expert's own top-two margin. These softmax values are NOT calibrated
probabilities or a guarantee of expected balanced-accuracy gain.

Keep all23 ports, the original five64-label supports and row-only query
exclusion. Use ONLY the primary5% admission cap, floor(.05*N_eval), unchanged
base fallback, observed support class guard, actual disagreement and positive
score. Ties use ascending globalrow. All scores are accessed before admission:
there is no compute-saving claim. No temperature/threshold/feature search,
portwise winner mixing, extra target labels or secondary-budget rescue.

Arms: advantage_bias, advantage_visual, frozen_bias and frozen_visual.
The frozen arms are the unchanged receiver-consistent gate policies. Primary
candidate is advantage_visual versus frozen_visual; advantage_bias versus
frozen_bias is the matched receiver diagnostic. Report BOTH advantage arms
against BOTH frozen arms, including negative results. A positive mean is only
a development screen, not independent confirmation or evidence of novelty.

Freeze all target scores, masks and predictions before reading query outcomes.
Report equal-port BA and accuracy, recovery/harm BA, admission counts, five
draw deltas, port wins/losses and paired port-bootstrap descriptive intervals.
The five draws overlap, so the statistical unit is the port, not115 contexts.
All data have been exposed to prior development. Stop this ranking branch
after this fixed screen; do not tune it against the target results.
