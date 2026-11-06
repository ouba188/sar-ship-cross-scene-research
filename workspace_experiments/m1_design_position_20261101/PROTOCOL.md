# M1 design x placement, OSS2, 2026-11-01

Reuse OSS2 source719 / historical development228 / fixed32 source references,
four classes, identity/MMSI/platform-orbit separation. No expanded test cohorts,
no Git, no formal-model overwrite. Reference E aggregate seed17 last-epoch4 is
the fixed carrier; the public carrier lacks private layer2[2] M2.

Two locations within layer3[5], both 256x10x10:
- pre: conv1/BN/ReLU -> M1 -> existing M2 -> conv2/BN/ReLU -> conv3/BN.
- post: conv1/BN/ReLU -> existing M2 -> conv2/BN/ReLU -> M1 -> conv3/BN.
Shortcut addition/ReLU then fixed M3 evidence update then layer4/head.

Three designs per position: existing additive mean/high/low context; bounded
class-context multiplicative modulation; same multiplicative modulation with
uniform mean context. One no-M1 control trains only conv3. The untouched carrier
is also reported. Both polarizations use separate M1 parameters, shared conv3.

Unlike the preceding output-only pilot, train all M1 reader/value/transforms/
output parameters, while fixing gamma to its pretrained value. Give the class
reader source-label auxiliary CE (.1), with missing polarizations excluded.
Also train the shared layer3[5].conv3 at lr1e-5 to allow downstream cooperation.
All other backbone convolutions, BN, M2, M3 and head stay frozen. No head-only
gain can explain arm differences; the no-M1 control measures conv3-only gains.

All M1 designs start with zero output and share the pretrained pre-location
reader/value/transforms. This tests practical warm-start relocation, not a
position-independent random-initialization architecture ranking. The post
reader must adapt to a different feature distribution under source supervision.
Mean control retains the same trainable parameter inventory, but its class
reader affects only the auxiliary loss, not the modulation context.

Seed17, four epochs120steps, batch24, identical source ordering and class/MMSI/
existing source-hard weights. AdamW wd.01, M1 output lr.001, other M1 lr.0001,
conv3 lr.00001, global gradient clip5. Final epoch only, no development-best
checkpoint, no LR/temperature/amplitude sweep. This is a feasibility pilot.
If a candidate improves on no-M1 and its same-position simple control, bounded
paired repetitions can assess whether that difference survives training order;
otherwise stop expansion and diagnose existing results without new designs.

Cache only pre-conv1-frozen-path input and shortcut, plus upstream low summaries
and fixed E evidence. M1 contexts are recomputed LIVE because the reader learns.
The E descriptors use layer1/2, so freezing its delta remains exact. Downstream
frozen layers retain autograd. Check original replay, zero-start parity across
positions, finite/nonzero live gradients and save/reload prediction parity.

Analysis: OA/BA, uncalibrated softmax NLL, paired rescue/harm, per-class results,
same-weight M1-off, uniform-context, spatial-mean and polarization diagnostics.
Inspect native VV/VH rescue/harm/stable examples with actual legal references.
Do not call a small result universal or claim new attention from multiplication.
Position scope motivated by existing bottleneck paths and previously read
CBAM/GCNet/GE/A2 comparisons; not an exhaustive placement/count search.
