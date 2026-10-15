# Source-Supervised Complement Compression

Completed 2026-10-08. Status: implemented and tested; not enabled because all
three datasets' mean full OA and BA are below the matched reference.

## Motivation And Implementation

The first projection-residual pilot recovered expert OA but did not consistently
improve BA. This follow-up tested whether source labels could turn the2560D
discarded-coordinate branch into an easier-to-adapt low-dimensional view.
It is a development follow-up, not an independently prespecified confirmation.

Using ONLY each existing identity-purged source cohort, fit balanced ridge1
to eight residual targets: +/-1 class target minus the original source expert
score. Inputs are the source-standardized PCA-complement coordinates plus a
penalized bias. The fitted2561x8 coefficient map produces eight coordinates,
standardized on source rows and divided by sqrt(8).

The target residual kernel is 0.5K_retained +0.5K_source_discriminant +1.
The64support ridge8, source prior, adapted base, inherited gate and5%handoff
rule stay unchanged. No source OOF interpretation is assigned to the fitted
discriminant representation. No target query labels enter the source fit,
support adaptation or routing. All three new policies are frozen before
query readout, with no parameter grid. Prior controls are copied unchanged.

## Results

Full model, mean percentages over five matched support draws:

| Dataset | Reference OA / BA | Source-guided OA / BA | Delta OA / BA pp |
| --- | ---: | ---: | ---: |
| F0921 | 29.633543 / 25.105149 | 28.816042 / 24.868668 | -0.817501 / -0.236482 |
| Public OSS1 | 43.066148 / 32.330577 | 42.661479 / 31.371816 | -0.404669 / -0.958761 |
| Public OSS2 | 32.382505 / 29.938364 | 31.735941 / 29.658824 | -0.646563 / -0.279540 |

Expert all-query OA changes -2.162123/-1.229572/-2.928552 pp on
F0921/OSS1/OSS2; expert BA changes -1.452997/-5.022368/-2.129991 pp.
The problem is therefore already visible in the receiving expert, not only
in the inherited handoff selector. On F0921 full OA and BA decline in all5draws;
the full model is even below its unchanged adapted base.

## Decision

Stop this source-eight-coordinate compression variant. Do not discard the
separate uncompressed complement OA gain, and do not infer that ALL source
knowledge is useless. This specific source residual projection did not transfer
effectively under the fixed recipe. More tuning was not run.

The result does not isolate whether dimension8, residual source supervision,
metric normalization or domain-specific source directions caused the loss.
It does establish that this implemented variant is not a useful upgrade.

Artifact files: `PROTOCOL.json`, `summary.json`, `metrics.csv`; each dataset
has `source_projection.npz`, frozen policies and detailed per-draw metrics.
Run via `oof_reliability_check/run_source_discriminant_pilot.py`. Earlier
models and all first-pilot policies remain unchanged.
Focused verification: the extended12-test suite passes, including invariance
to source-external labels/features, sensitivity to source labels, and equality
of the guided kernel with its explicit feature map and single bias.
