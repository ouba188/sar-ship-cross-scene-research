# Omitted visual-feature residual: fixed screen fails

## Decision

Do not promote the added visual tail. It slightly improves the full-query
expert's mean BA over archived linear, but fails the matched-energy control
and loses at the primary5% expert handoff budget. No parameter sweep follows.

Output `../oof_visual_tail_results_20261007`, protocol
`VISUAL_TAIL_PROTOCOL_20261007.md`.23 ports x5 supports,64 labels each;
46 source decoder fits,230 target fits,1035 policy metrics,73.13seconds.
No new encoder/PCA/original-head fits or target labels. Raw cache receipts
match the established OOF inputs, replacing costly PCA/head reconstruction.

## Results: equal-port BA (%)

| Arm | Full-query expert | 5% handoff |
| --- | ---: | ---: |
| Archived linear | 52.894867 | 45.846390 |
| Omitted-feature visual tail | 52.997162 | 45.523331 |
| No-new-information energy control | 53.073001 | 45.657858 |
| Archived bias expert | 51.756774 | 45.802450 |
| Previous advantage whole policy | not a new expert | 45.885986 |

At5%, tail-minus-linear is-0.323060pp:9 positive/14 negative ports,
descriptive port-bootstrap95% interval[-0.796337,-0.015524]pp. Four of five
draw means are negative. Tail-minus-energy is-0.134527pp and tail-minus-old
advantage is-0.362655pp. Mean admission fraction is identical4.932516%.

Full-query tail-minus-linear is+0.102295pp (15positive/8negative), but
tail-minus-energy is-0.075840pp (10positive/13negative). The energy control
alone adds0.178135pp over linear. Thus this study does not demonstrate a
BA benefit specifically attributable to the extra visual information.
Full-query tail accuracy60.554862% exceeds linear58.425765%, but this is a
secondary metric and cannot rescue the BA/budget primary screen.

## Actual information change

Reuse the complete VV2048 and VH512 frozen embeddings. For each source
fold/polarization, least-squares regress centered raw features on the existing
retained128/64 coordinates. Append its residual after matching each branch's
source energy. The source-sample residual is orthogonal to retained columns;
it is NOT claimed an exact pointwise PCA complement. The pre-run draft
inverse-decoder check was removed for this mathematical reason, not because
of a target result. Source fitting uses no query statistics or labels.

The candidate has192+2560 visual coordinates; its matched-energy control
uses onlysqrt2-scaled existing192 features. Both retain the same penalized
intercept, source classifier prior,64 support labels and ridge strength8.
Energy matching does not match every spectral/capacity property. Related
older residual/Fisher and learned32D experiments were found, but this exact
comparison was not found in the bounded historical search. It is not claimed
to be a new residual learning principle.

## Verification and limits

Five focused tests passed: exact-PCA reference residual, general retained
projection against independent least squares, source-only parameter fitting,
weighted-primal Ridge equivalence and correction from information absent in
retained features. Independent replay passed:230 weighted fits,460 gate/mask
replays,345 unchanged budget controls,1035 metric rows. Source decoders refit
independently by QR on two fixed ports, with all46 query-tail blocks replayed.
Maximum score error2.22e-15 and gate error5.89e-15. Output `VERIFICATION.json`
also preserves the initial verification failure: near-zero source centers
differed after a precision-order mismatch. Explicit float64 promotion AFTER
float32 standardization reproduces NumPy2.x producer semantics; no experiment
parameters or outcomes were changed and no tolerances were relaxed to pass.

All targets remain development-exposed. Support/query is row-only; source
purges are inherited. Both old and new expert scores are computed before
admission, not a demonstrated5% compute budget. No main model or manuscript
was changed. A subsequent adapted-BASE study is separately declared in
`BASE_ADAPT_PROTOCOL_20261007.md`; it changes the deployment setting and
does not rescue this negative fixed-base screen.
