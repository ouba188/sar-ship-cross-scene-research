# Receiver-consistent application of a frozen reliability gate

Declared after the matched-mask budget study, before this candidate readout.
The matched5% screen remains false; its results are not overwritten. This
follow-up changes the gate INPUT, not its coefficients, threshold or budget.

Reuse original source-OOF four-feature LR parameters from each port's trusted
local context-run source_model.pkl/prior. For each receiving expert, recompute
expert margin/entropy using THAT expert's scores; base margin/entropy remain
unchanged. Apply the original source center/scale and LR coefficients. Require
original fusion score/policy reproduction. This is uncalibrated transfer of a
frozen gate to an adapted predictor, not OOF-trained reliability for that new
predictor and not a novel gate-learning algorithm.

Four receivers: unchanged fusion, support-observed-class restricted fusion,
anchored bias and anchored visual. For restricted fusion compute top-two gap
and softmax entropy over ONLY observed classes; keep all eight output ids.
An input-only inventory confirms all115 supports have at least3 observed
classes, so top-two gaps exist; do not invent a one-class confidence rule.

For each receiver, eligible = evaluation & (base!=receiver_argmax) &
observed[receiver_argmax] & (recomputed_LRscore>0). Rank descending score,
stable ascending globalrow tie-break, take at most floor(B*n_evaluation).
B=.05 primary, .10/.20 secondary. Supports remain64 and unchanged. At fixed
budget, masks may now DIFFER across receivers; report admissions and changes.
This is a whole-policy comparison, not the preceding same-mask attribution.

All scores/masks/predictions for23ports/5draws freeze before query outcome
readout. No fitting or hyperparameter selection. Primary contrasts at5%:
consistent visual minus stale-mask visual; consistent visual minus consistent
bias. Also retain same-budget original fusion and observed-class controls.
Prespecified screen passes only when both primary mean BA differences are
positive. Report all budgets without replacing the primary by a favorable one.

Every receiver's gate now accesses its query scores before admission. Admission
caps do NOT establish selective feature inference, compute savings or latency.
Keep the source/target split and historically exposed development limitations
of the parent study. In particular, support/query remains row-only.

If frozen-gate input consistency is insufficient, the next substantive route
would require new source episode or support-cross-fitted reliability training,
not further gate thresholds, coefficient scaling or budget sweeps here.
