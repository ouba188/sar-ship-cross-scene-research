# Mean-anchored contrast projection: targeted design test

Added after the completed 14-fit position/parameter search. Development best class OA/BA59.6491/52.6473 versus tuned mean60.0877/52.9282. Class updates are already8.4/8.6% of input norm; more gain is not an adequate explanation. Same-weight uniform-context changes no top1 in the selected post model. This motivates a test of content organization, but does NOT prove shared projection is the root cause.

Keep existing mean/high/low reads. Define a common mean context b and differential context d from high/low tokens minus the mean token. Use separate zero-initialized output projections for b and d, then tanh channel modulation. All original M1 reader/value/three transforms remain trainable; no new class labels, input regions, frequency branches or reference information.

Class: x' = x + gamma*x*tanh(W_b*b + W_d*d).
Matched mean control: x' = x + gamma*x*tanh(W_b*b + W_d*b).

Both have identical parameter counts, gains, initialization, training schedules and optimization groups. A second mean projection is algebraically redundant but gives the control equal parameterization/optimization opportunity. This is a targeted representation test, NOT proof of strong originality.

Four fits: pre/post conv2 x class/mean; same task_lr config from initial search, source-only calibrated readers, six epochs/180steps/seed17 each. Select development BA/OA/NLL (including epoch0). Report independently selected and fixed-final results plus same-weight differential-off and spatial-mean. Preserve the original14 runs and compare their independently tuned best too.

If a useful advantage appears, freeze config/epoch and repeat orders29/43. If not, report the negative result and actual remaining uncertainty instead of unlimited modifications. Native VV/VH/query-reference review is mandatory.
