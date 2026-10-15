# Residual expert reconnected to budgeted handoff

Date: 2026-10-07. Two sequential fixed development experiments, zero new fits.
Both retain23ports, the same five support draws/port and64 labels/draw. Labels
in support are excluded from both deployment admission and evaluation. The
budget is a strict admission cap floor(B*n_evaluation), not compute measured.

## Decision

Prioritize the adapted expert with receiver-consistent gate inputs as a
budgeted development candidate. Do not merely replace the expert's output
under the old masks. The strongest current evidence concerns the combined
adaptation/input-consistency pipeline; a distinct large visual-residual gain
over simple bias adaptation is NOT established at small budgets.

The first matched-mask primary screen remains FALSE. The separately declared
receiver-consistency screen is TRUE under its positive-mean criteria. Do not
merge these studies into an apparent first-try successful preregistration.

## First study: identical admission masks

Protocol: BUDGETED_RESIDUAL_PROTOCOL_20261007.md.
Run: ../oof_budgeted_residual_results_20261007.
Original four-feature source-OOF LR0 ranking is capped without changing its
inputs, eligible old-model disagreements or threshold. All expert candidates
receive exactly the same admitted rows and cost. Only expert predictions differ.

| Maximum admission budget | Old fusion BA, % | Bias expert BA, % | Visual expert BA, % |
|---|---:|---:|---:|
| 5% (primary) | 45.0657 | 45.1287 | 45.1272 |
| 10% | 46.1182 | 46.3335 | 46.3582 |
| 20% | 47.7079 | 47.9453 | 48.0874 |

At5%, visual-minus-fusion is +0.061492 pp,10 positive/9 negative/4 tied ports,
descriptive port-bootstrap interval[-0.041717,0.189596]. Visual-minus-bias is
-0.001478 pp,9 positive/10 negative/4 tied ports, interval[-0.060247,0.048305].
The primary screen therefore fails; the tiny difference from bias does not
establish superiority or meaningful inferiority. No threshold was tuned.

Observed-class-restricted fusion is identical to fusion under these masks,
because admission already requires the original fusion argmax class observed.
This is an expected control identity, not a failure of the comparison code.

The prespecified base-uncertainty selector gives5% visual BA44.2513%, versus
fusion43.5141%, bias44.1512%, restricted fusion44.1360%. Its visual-minus-fusion
gain is+.737162pp (21/23 positive), but its absolute BA is worse than the capped
old gate. Do not select this weaker selector to inflate the claimed gain.

## Second study: actual receiver scores enter the gate

Protocol: RECEIVER_CONSISTENT_PROTOCOL_20261007.md, declared after the first
study and before this readout. Run: ../oof_receiver_consistent_results_20261007.
Keep original source gate coefficients, feature center/scale and threshold0.
Recompute expert margin/entropy and disagreement using the actual receiver.
Use the same support-observed-class guard on that receiver's argmax. No gate
fit, new labels, threshold scan or expert parameter changes are performed.

| Maximum admission budget | Old fusion pipeline BA, % | Consistent bias BA, % | Consistent visual BA, % | Visual minus old, pp |
|---|---:|---:|---:|---:|
| 5% (primary) | 45.0657 | 45.8024 | **45.8464** | **+0.7807** |
| 10% | 46.1182 | 47.3958 | **47.4920** | +1.3738 |
| 20% | 47.7079 | 49.3523 | **49.5450** | +1.8371 |

At5%, versus unchanged fusion pipeline: +0.780675pp,17/23 positive,
interval[0.274702,1.314875]. All five draw-level mean gains are positive
(0.5887 to1.0038pp). These overlapping draws are NOT independent replications.

Primary consistent-visual minus stale-mask visual: +0.719183pp,18/23 positive,
interval[0.265736,1.188002]. This validates input consistency as a useful
whole-policy change in this development setting, not a newly invented method.

Primary consistent-visual minus consistent-bias: only +0.043940pp,
11/23 positive and12 negative, interval[-0.294968,0.383164]. Two of five
draw-level means are negative. The declared mean-only screen passes, but the
visual component alone is not robustly superior. At10%/20% the extra visual
gain over bias is+.096212/+.192696pp, with both intervals also including zero.

Consistent restricted-fusion BA is44.1934/44.5797/45.2639% at5/10/20%.
Deleting unobserved classes changes confidence distributions; this makes the
frozen source reliability gate an imperfect interface for that control too.
Do not call the comparison proof that hard class restriction is universally bad.

## Cost and interpretation

Mean actual admission rates at5%: oldfusion4.9019%, consistentbias4.9325%,
consistentvisual4.9325%. At10%:9.6362/9.9246/9.9246%; at20%:
17.0134/19.3296/19.3760%. All respect each context's integer cap. The second
study matches the budget ceiling, not exact admitted counts or sample identities;
some gain may reflect using otherwise unspent capacity. Exact same-mask expert
attribution is the FIRST study and is much smaller.

In the first study admitted rows are charged even when the new expert returns
the base prediction; actual changes are separately saved. In the second study
eligibility uses the receiving expert's actual disagreement, so admitted rows
and changed rows coincide. No artificial refill or omitted cost is used.

The confidence gates read complete receiver outputs before selecting rows.
Therefore this is an admission/replacement-budget result, NOT a measured
5%-expert-computation,95%-compute-saving or20x-speedup result. The64-label
adaptation overhead is unchanged and separate. No backbone inference or timing
benchmark was performed. Prior full-query visual BA52.8949% remains a different
operating point; it is not the5% result.

Mombasa still loses1.5837pp against the old5% pipeline and2.4622pp against
consistentbias. Five other ports also lose against the old pipeline. No port
was dropped or assigned a hindsight-selected receiver. Source/target splits
are inherited; target support/query isolation is row-only. All data remain
historically exposed development data. The frozen gate is not retrained or
calibrated OOF reliability for the adapted expert distribution.

## Implementation and verification

- First core/runner: budgeted_residual_handoff.py, run_budgeted_residual_handoff.py.
- Second core/runner: receiver_consistent_handoff.py, run_receiver_consistent_handoff.py.
- Five budget tests and three confidence-interface tests passed.
- Each study saved all23-port policies before its own candidate readout.
- First run2.85s,2760 policy/metric rows,690 shared masks,0fits.
- Second run3.13s,1380 new policy/metric rows plus345 inherited stale controls,0fits.
- Original fusion scores/masks are reproduced in the receiver-consistent run.
- Both independent verifiers completed with exit0; receipts are in each run's
  VERIFICATION.json. First:2760 metric rows,690 shared masks and18 contrasts.
  Second:1725 rows (1380 new,345 inherited),12 contrasts; all345 fusion controls
  unchanged, maximum gate-score replay error5.77e-15. Source priors, support
  exclusion, confidence, order/ties, caps/costs and confusion metrics replayed.
  Replaying saved artifacts does not independently certify historical readout timing.

## Next useful work

Keep the simple consistent-bias pipeline as a strong default comparator.
The concrete next mechanism is reliability training matched to the adapted
expert: source-held-port support/query episodes or strictly support-cross-fitted
labels, so the router estimates incremental benefit from the NEW receiver.
It must not train on its own in-sample residual successes or target query truth.
Retain5% as primary and distinguish admission budgets from feature-access cost.
No further threshold/budget search, generic attention or facility concatenation
is justified by these findings.
