# Bounded facility pilot: complete development readout

Date: 2026-10-07. Run: ../oof_context_results_20261007.
Fixed protocol: CONTEXT_PROTOCOL_20261007.md.

## Decision

Do not promote direct concatenation of the audited facility descriptors into
this source-only logistic gate. Both prespecified primary contrasts are
negative. Do not sweep regularization, select favorable ports, or call this
an effective innovation module. This does NOT refute all use of port knowledge.

The current pipeline now has real aligned local facility inputs, a functioning
missing-data fallback, and matched coverage/content controls. These are usable
research infrastructure, not evidence of a successful new model.

## Results

Equal-port BA after averaging five support draws within each of 23 ports:

| Arm | BA, % | Difference from original LR, pp |
|---|---:|---:|
| Original four-feature OOF LR | 48.0408 | 0 |
| Covered-source confidence only | 48.0573 | +0.0166 |
| Covered-source confidence + known masks | 47.6669 | -0.3739 |
| Covered-source confidence + masks + values | 47.3098 | -0.7310 |
| Original simple gate | 47.9034 | -0.1374 |

Facility values minus mask-only control: -0.357080 pp, 2 positive and 12
negative ports (9 ties); descriptive port-bootstrap 95% interval
[-0.639103, -0.132068] pp. Facility values minus original LR: -0.730983 pp,
2 positive and 15 negative ports (6 ties), interval [-1.586520, -0.204370] pp.
Both contrasts are negative in every support draw. These overlapping draws
are not independent replications.

Covered-confidence versus original LR is only +0.016576 pp, 6 positive and
7 negative ports, interval [-0.025539, +0.065580] pp. This arm uses no facility
values: its small positive mean must not be credited to facility semantics.
No claim of statistical significance or reliable improvement is made for it.

All comparisons use full query sets excluding the same 64 support rows.
Supported-subset diagnostics are saved per port/draw in metrics.json and CSV;
they have different class supports and are not substitutes for full-port BA.

## Inputs and Limits

The immutable audit exports 38,091 x 60 values/known, with 18,098 rows having
bounded support; 18,092 supported rows belong to active target ports before
support exclusion. All rows are retained; unusable context has exact original
LR score and prediction fallback. Coverage is highly uneven across ports.
No historical broader support masks or target outcome fields enter the gate.

These are mapped local facility proximity/contrast descriptors, not verified
berth functions, actual occupancy or AIS trajectories. Some coordinates are
recorded proxies/inherited centers. Maps from 2025 and 2026 are not temporally
certified against SAR acquisitions. Source OOF partitioning is retained;
target support/query isolation remains row-only. This is uncapped selection,
not the prior 5% replacement-budget study. The 64 labels still supply the
observed-expert-class guard; no target coefficient adaptation is performed.
All ports were historically used in development, not independent confirmation.

## Verification and Cost

- Four focused tests passed: known zero versus missing, unobserved source
  features, masked source scaling, exact fallback, and support-class guard.
- All 23 port policies and coefficients were saved before candidate readout.
- Original LR/base/fusion/simple/s2/oracle metrics reproduce with error zero.
- Independent confusion-matrix replay: 115 contexts, 5,175 metrics, maximum
  error 2.22e-16, including score-to-prediction and missing-fallback checks.
- 69 new covered gate fits and 23 reproduced original priors; no head or
  backbone retraining. Run completed in 76.83 seconds, exit 0.

## Research Implication

Richer output features and direct facility concatenation have now both failed
their fixed tests. Preserve the original LR and simple baselines. A future
context proposal should test a different mechanism, such as whether local
evidence supports the particular base/expert disagreement, with source-held-
port evidence controlling when to trust the correction. That is a hypothesis,
not a demonstrated improvement or a literature-verified novel contribution.
Do not launch a broad architecture search from these target outcomes.
