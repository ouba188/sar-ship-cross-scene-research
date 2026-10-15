# First Paired Spatial Information Pilot

## Outcome

The new expert really receives VV spatial information and learns a nonzero
competitor-directed contrast. However, this first unconstrained implementation
does NOT improve over the already retained aligned-global candidate or the
matched ordinary paired-pooling control. Do not use a positive comparison
against the older frozen model to claim that the attention term itself works.

Same corrected F0921 cohort,64labels,5%handoff, five support draws:

| Configuration | Full OA | Full BA |
| --- | ---: | ---: |
| Old frozen replay | 29.634353 | 25.106684 |
| Previous aligned-global | 29.326474 | 25.435555 |
| New paired pooling | 28.958639 | 25.355954 |
| New competitor-directed contrast | 28.853636 | 25.309197 |
| Same contrast model, spatial term disabled | 28.938870 | 25.330872 |

Contrast minus matched paired pooling: -0.105003ppOA/-0.046757ppBA.
Contrast minus aligned-global: -0.472838ppOA/-0.126359ppBA.
Contrast minus its inference ablation: -0.085234ppOA/-0.021675ppBA.
The global-pooling branch's +0.249269ppBA versus old frozen replay is retained
as an observed positive comparison, but it does not beat aligned-global.

## Mechanism And Diagnosis

Read MODULE_DESIGN.md for the implemented formulas and prior-art boundary.
24focused module tests passed, including zero-initialization parity, paired
input gradients, class-direction derivatives, and64support backpropagation.
The source-only episodic training uses the actual retained residual expert,
not an auxiliary classifier. Both complete six-epoch runs finished before
query readout; original base predictions, supports, queries and budgets match.

The spatial contrast is not inactive: across123489target rows, mean contrast
norm is6.399334 and disabling it changes final expert features by meanL2
11.160635. Mean learned residual norm29.054363 is comparable to original
VHfeature norm27.396922. This shows substantial feature alteration, not gain.

Full-query expert diagnostics, outside the5%budget:

| Configuration | Expert OA | Expert BA |
| --- | ---: | ---: |
| Previous aligned-global | 29.802714 | 27.036078 |
| Paired pooling | 25.913551 | 24.850372 |
| Competitor contrast | 25.392749 | 23.642100 |
| Same contrast, spatial term disabled | 26.179785 | 24.241234 |

Thus the negative result is not explained solely by the old handoff gate.
Large new feature shifts and source-head damage are a concrete follow-up
hypothesis, not a proven unique cause. The next bounded structural follow-up
freezes the original VH encoder and restricts new evidence to directions
invisible to source prior logits but available to64support adaptation:
../support_visible_polar_20261008. Keep this failed precursor intact.

This remains historical F0921 development, not independent confirmation.
Dual-polarization attention already exists in prior literature; implementation
and changed information alone are not evidence of a publishable new mechanism.
