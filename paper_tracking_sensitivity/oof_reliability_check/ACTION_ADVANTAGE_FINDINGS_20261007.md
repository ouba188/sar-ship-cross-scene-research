# Same-head action advantage: tiny positive mean, not a breakthrough

## Decision

Retain this simple control, but do not replace the main method or claim a
robust new module from its0.039596pp gain. End this fixed ranking branch
without a temperature, threshold or feature sweep.

Protocol: `ACTION_ADVANTAGE_PROTOCOL_20261007.md`.
Output: `../oof_action_advantage_results_20261007`.
23 ports, five original64-label supports each, only the primary5% cap,
460 policy metric rows and0 new fits. No new data, labels or preprocessing.

## Equal-port Results (%)

| Policy | BA | Accuracy | Recovery BA | Harm BA |
| --- | ---: | ---: | ---: | ---: |
| Same-head advantage, bias expert | 45.798026 | 46.489436 | 3.235386 | 0.706355 |
| Same-head advantage, visual expert | 45.885986 | 46.899883 | 3.252652 | 0.635662 |
| Frozen gate, bias expert | 45.802450 | 46.525871 | 3.272799 | 0.739345 |
| Frozen gate, visual expert | 45.846390 | 46.926903 | 3.346132 | 0.768737 |

All arms have identical4.932516% mean admission fraction, reflecting the
integer5% cap. Recovery/harm BA are measured relative to the unchanged base.

Primary advantage_visual minus frozen_visual:
- BA +0.039596 percentage points; accuracy -0.027021pp.
- 11 positive and12 negative ports; descriptive port-bootstrap95% interval
  [-0.202156,0.328396]pp.
- Five support-draw deltas: +0.081465,-0.120365,+0.454525,-0.141843,-0.075804pp.
- Recovery BA decreases0.093480pp while harm BA decreases0.133075pp.
  The small net mean gain is NOT from recovering more cases.
- Mombasa improves2.373010pp; Newcastle loses1.160709pp, New York loses
  0.686259pp and Callao loses0.544182pp. No portwise policy mixing is used.

Matched bias comparison is negative: advantage_bias minus frozen_bias
-0.004424pp,10 positive/13 negative ports. Advantage_visual minus frozen_bias
is+0.083536pp,11 positive/12 negative ports with an interval spanning zero.
The mean-only primary development screen passes by definition, but this is
not evidence of reliable superiority across ports or support draws.

## Mechanism And Interpretation

Score = softmax(adapted expert)[expert argmax]
      - softmax(adapted expert)[base argmax], temperature1.

Both terms come from the actual receiving expert. This is not the gap
between two models' maxima and not necessarily the expert's top-two gap.
The same receiver predictions and observed-support-class guard are retained;
only ranking changes. These softmax scores are not calibrated probabilities.
The historical conditional-likelihood posterior difference is a related
existing baseline, so no new action-advantage principle is claimed.

## Verification And Next Direction

Three focused tests passed: actual action-pair comparison, numerical shift
invariance/agreement, and budget exclusion/global-row tie ordering. The first
test fixture was corrected before the experiment because its original
numbers did not create the intended ranking reversal; model code was unchanged.
Independent verification completed: all23 ports,460 metric rows,230
candidate policies and230 unchanged controls passed; maximum score replay
error0. Output `VERIFICATION.json` is the verifier's saved machine receipt.

All targets are historically exposed development data. Support/query is
row-disjoint, not product/vessel-disjoint. All receiver scores are available
before admission: this is not a5% compute budget or a speedup result.

Retain the established frozen visual policy as the main reference and this
candidate as a low-complexity comparator. The next proposed substantive
direction is improving the expert's classification on difficult cross-port
examples, especially cases where both current heads are wrong, rather than
adding another confidence-only gate. Any such change needs its own fixed
source-selected protocol and a matched expert control; this run establishes
neither its gain nor its novelty. No further experiment is silently queued.
