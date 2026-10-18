# Pre-pooling local evidence probe

2026-10-18, fixed before running this round. Same 84 provisional associations,
seven raw classes, five acquisitions, B16 source supports and five support draws.
No target selection, new neural training, label changes, or coefficient sweep.

Hypothesis: regional responses need to be read separately BEFORE local average
pooling, rather than modulating an already mixed vector by region proportions.
Read native evidence first: previous row71 isolated small target versus its
same-class row82 support beside large bright structures motivates this test.

Frozen ImageNet R50 layer1 features, identical paired VV/VH128 inputs. The full
image still passes through the same backbone; this is feature-map readout, not
pixel intervention or pure physical separation (receptive fields cross borders).

Matched 12-token / 3072-dimensional readouts per polarization:
- grid12: ordinary 3x4 spatial pooling over all pixels.
- postpool: 2x2 quadrants pooled first, each vector repeated into target, other
  candidate, and remaining-background slots; same availability as prepool.
- prepool: within each of the 2x2 quadrants pool target, other candidate and
  remaining background separately. No water/nonwater map distinction.

Each token is centered using only support images where that token exists. A
token absent in all supports is disabled for the episode. Missing query tokens
remain zero. Normalize each nonzero centered token, then the whole descriptor.
This gives present tokens equal initial norm rather than letting large background
area dominate; grid12 uses exactly the same rule. Record tiny token areas and
support coverage rather than selecting a threshold from outcomes.

All new arms use 0.5 global + 0.25 retained old2x2 + 0.25 new kernel. Compare to
the unchanged 0.5 global + 0.5 old2x2 baseline as well as each other. Scalar norms
and directions may change; if risk improves without discrimination, use the
existing label-free norm-matched diagnostic. Do not interpret same-token capacity
as proving all statistical effects of availability are controlled; postpool has
the exact same missing-token organization to address that particular issue.

Same mean-support classifier, softmax temperature0.1 and metrics as previous
rounds; report equal-acquisition and query-weighted views, paired rescue/harm,
per-class and per-acquisition results, actual native chips and actual supports.
Preserve source association uncertainty and tiny 45FE single-class query fold.
These are relationship probes, not the complete dual-backbone model main table.
