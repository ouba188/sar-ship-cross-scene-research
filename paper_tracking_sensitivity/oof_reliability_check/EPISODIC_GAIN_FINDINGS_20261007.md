# Adapted-expert episodic gate: completed negative screen

## Decision

Do not promote the source-episodic gate. At the prespecified primary5% cap,
neither matched binary training nor three-outcome expected-gain training
improves the existing receiver-consistent policy. Keep the original outputs
and do not tune these models against the target results.

Output: `../oof_episodic_gain_results_20261007`.
Protocol: `EPISODIC_GAIN_PROTOCOL_20261007.md`.
23 outer ports, five64-label supports each,2530 source adaptation episodes,
46 source gate fits,0 new target/head/backbone/PCA fits,85.93 seconds.
All policies saved before this run's query readout; this timing follows the
runner's execution, not an independently timestamp-certified preregistration.

## Equal-port BA (%)

| Admission cap | Matched binary | Three-outcome gain | Frozen bias | Frozen visual |
| --- | ---: | ---: | ---: | ---: |
| 5% (primary) | 45.709694 | 45.633952 | 45.802450 | 45.846390 |
| 10% | 47.231142 | 47.336039 | 47.395780 | 47.491993 |
| 20% | 49.573626 | 49.736642 | 49.352282 | 49.544978 |

At5%, gain minus frozen visual is -0.212439 percentage points:8 positive
and15 negative ports, descriptive port-bootstrap95% interval[-0.660870,
0.190295]. Gain minus matched binary is -0.075743pp. Binary minus frozen
visual is -0.136696pp. All four arms have identical mean admission fraction
4.932516% at this operating point (integer cap rounding).

At20%, gain exceeds frozen visual by0.191664pp, but only9/23 ports improve;
the interval spans zero and accuracy decreases1.922363pp. This secondary
result does NOT rescue the negative primary screen.

## What Was Tested

Every outer target is excluded from source gate training. Genuine nested
source OOF base/expert heads are reused. Within each held source port,
five64-label supports adapt the exact bias-only residual receiver; remaining
rows train four-confidence-feature logistic gates. Matched binary excludes
both-wrong neutral cases; three-outcome gain retains them and ranks by
P(recovery)-P(harm). Equal source-port/episode/true-class weights and fixed
C=1 are used. Target receivers and support labels are unchanged.

This result tests these four-feature linear gates, NOT every possible
episodic learner and NOT the usefulness of cross-port knowledge in general.
The structural distinction is real, but a new objective alone did not yield
better budgeted prediction here. No top-journal novelty claim follows.

## Verification And Limits

The20 focused receiver/gate tests passed. The independent agent completed
source-bank, scoring, selection and metric replay:2530 episodes and1380
metric rows passed. Output `VERIFICATION.json` records that completion
message; the episodic verifier printed, but did not itself save, its detailed
receipt. No unreported numerical error bound is inferred from the message.

Source head training retains inherited port/MMSI/product purges. Episode
support/query and target support/query are row-disjoint ONLY. All targets
have historical development exposure; five overlapping support draws do not
constitute115 independent datasets. Budget is an admission/replacement cap:
all expert scores are available before selection, with no measured compute
savings. No manuscript claim was upgraded based on this negative result.

## Bounded Follow-up

A separately declared training-free action-pair ranking control was run after
this result. See `ACTION_ADVANTAGE_FINDINGS_20261007.md`. It does not change
this study's failed primary screen.
