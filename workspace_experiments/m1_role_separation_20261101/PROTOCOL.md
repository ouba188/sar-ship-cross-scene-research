# M1 role-separation pilot, 2026-11-01

Question: can M1 modulate existing local responses, leaving region organization
to M2 and reference-conditioned evidence to M3, without losing useful accuracy?
Changing addition to multiplication alone is not a novelty claim.

Frozen anchor: OSS2 public four-class model plus source-hard E aggregate M3,
seed17, last epoch4. This is a historical development scaffold, not a replacement
of the private formal model (in particular, no private layer2[2] insertion).
719 source observations, 228 development observations, fixed32 source references.
Source/development IDs, MMSIs and platform-orbits are disjoint. No expanded
selection/evaluation predictions are used in this pilot. Development is not blind.

Four arms, identical zero output initialization, source order, fixed references,
four epochs/120 steps, batch24, seed17, AdamW lr0.001/wd0.01, clip5.
Only two 32-to-256 M1 output projections train. Reader/value/transforms remain
the same supervised pretrained parameters. M2, M3, backbone, head and BN freeze.
Source class/MMSI weights and existing source-hardness weight2 are held constant.
The final epoch is reported; no development-best selection or parameter sweep.

Let c be the existing mean/high/low class-response context, z=Wc, g the frozen
original per-polarization gamma, s the source-only valid-pixel channel RMS:

- additive: x + g*z
- bounded_add: x + g*s*tanh(z)
- class_gate: x + g*x*tanh(z)
- mean_gate: same multiplicative update, replacing high/low weights by uniform
  valid-position weights both when reading and redistributing context.

The mean arm has the same stored machinery and trainable parameter count, but
its constrained contexts have less expressive variation. This is an information
ablation, not proof of superiority to every generic attention block.

All arms start with M1 bypassed; retain the untouched pretrained M1/E system as
an additional anchor. Source RMS is measured before training and never uses
development labels. Cache only quantities upstream of trainable M1, fixed
contexts and fixed E updates. The frozen downstream path remains differentiable.
Replay untouched M1 against the historical wrapper before training, then check
identity initialization and actual gradient flow. Cached E updates are exact
because E descriptors and write masks use layer1/layer2 before this M1 insertion.

Readouts: OA/BA/NLL, source loss, rescue/harm counts, per-class outcomes,
actual update norms, spatially averaged gate and VV/VH-only gate diagnostics.
Inspect native VV/VH query/reference chips for rescue, harm, stable correct/wrong.
One seed is feasibility only, not repeatability. No Git or formal-model edits.

Literature boundary: CBAM (ECCV2018), Gather-Excite (NeurIPS2018), GCNet (2019),
Double Attention (NeurIPS2018) already cover context gathering, multiplication,
or gather/distribute residual updates. A useful candidate contribution requires
evidence for class-conditioned response differences, not a new operator name.
