# Spatial ownership probe, frozen before evaluation

2026-10-18. Continue the preceding 84-object exploratory cohort without changing
labels, images, features, queries, or source support draws. Identity associations
remain spatial-containment proposals. This is not the full dual-backbone model.

Question: does object ownership improve cross-acquisition support relationships
when the already useful spatial grid representation is preserved?

Use the cached layer1 2x2 grid descriptors and final global descriptors. For each
cell retain the image vector and compute target/other-candidate/rest fractions.
Rest combines previous water and nonwater/unknown, so this experiment does not
use the map-water distinction. Detection masks are inputs, not verified segments.

Center each image descriptor using the same support set and normalize each
polarization exactly as before. Only then attach sqrt(ownership fractions) to
each spatial cell. This lifted feature preserves the original descriptor norm;
absent ownership channels remain zero and are never centered afterward.

The resulting inner product multiplies each corresponding-cell feature product
by sum_r sqrt(p_query,r * p_support,r). It changes relation compatibility, not the
image features. This is a fixed, testable encoding, not learned attention.

Arms fixed in advance:
- global: existing final-feature kernel G.
- global_grid: 0.5 G + 0.5 L, exact preceding baseline.
- spatial_ownership2: 0.5 G + 0.25 L + 0.25 O2 (target versus rest).
- spatial_ownership3: 0.5 G + 0.25 L + 0.25 O3 (target, other candidates, rest).
- ownership_only: 0.5 G + 0.25 L + 0.25 A3, replace the modulated image term with
  the ownership-only kernel to test whether mere box composition suffices.

O2/O3 remain positive-semidefinite feature kernels and can later enter the same
support solve. This is an integration property, NOT evidence of full-system gain.
Output dimensions differ (2048/3072 local lift versus 1024 original), but there
are no new learned parameters, training steps, or new backbone inference.

No weight search, new data selection, label correction, or extra neural training.
Seven-class mean-support classifier and temperature0.1 remain unchanged. B16,
five support seeds, leave-one-acquisition out. Report equal-acquisition means,
query-weighted means, paired support-draw variability and per-class effects.
Keep retrieval/risk improvements separate from OA/BA. Trace representative native
VV/VH and actual supports including persistent correct/wrong outcomes. Test the
specific risks: baseline replay, missing blocks, aligned cell order, normalization
and query information not changing support-only fit. No held-out-label selection.
