# Bounded source-reader repair, declared before the repair fits

Initial7 fits completed. Same-position class_gate and mean_gate have identical
top1; no reason to expand seeds. However, inherited class-reader semantics are
not well matched to the post location: weighted source auxiliary CE epoch1/4
is3.975/2.905 post versus1.117/1.095 pre. This is an observed initialization/
adaptation mismatch, not proof that it causes the missing classification gain.

Correction: calibrate the linear class reader separately on each position's
frozen SOURCE pooled activations, with the same source719 weighted auxiliary
CE, AdamW lr.001 wd.01, exactly200 full-batch steps. No development data or
labels are used to fit or select the readers. Keep all value/transforms weights
from the original warm-start and output zero. Then repeat only four arms:
pre/post x class_gate/mean_gate, same original4epochs120steps seed17.

The original no-M1 control remains valid: it has no changed reader on the task
path and its conv3 training inputs/schedule are unchanged. No additional
position, gain, loss coefficient, or epoch selection. Preserve original7 fits.
Stop this repair round if class-conditioned gain still fails to exceed its
same-position mean control. Successful reader calibration by itself is not a
classification gain and does not prove the class-dependent spatial mechanism.
