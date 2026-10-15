# Post-readout class-support control for the adapted-base setting

Declared2026-10-07 AFTER the main base-adaptation result, BEFORE this control's
query readout. The large bias-adaptation gain motivates one explanatory
baseline, not a parameter search or a change to the original primary screen.

For each original64-label support, mark classes that actually occur in it.
Restrict the unchanged source BASE logits to those columns and take argmax.
No score fitting, new target labels, source retraining or expert handoffs.
This permits global changes from the original base, like the newly adapted
base setting. It does NOT assert that unobserved classes are absent in reality.

Same23ports,5supports,full query-minus-support evaluation. Freeze every
prediction before query readout. Compare with original_base, base_bias,
base_visual and visual_handoff from the unchanged main run. Also report
visual_handoff minus this control as a stronger-comparator diagnostic.
If this simple control explains the improvement, do not attribute it to a
new adaptation mechanism. Preserve the main result and explicitly label this
follow-up post-readout. No second coverage variant or extra gate is tested.

Output: oof_base_adapt_results_20261007/coverage_control. Same development,
row-only isolation, and admission-not-compute limitations as the parent.
