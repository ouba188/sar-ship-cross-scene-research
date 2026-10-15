# Source-metric local residual expert: fixed development screen

Written2026-10-07 before candidate target readout. The previous confidence
gate branch is closed. This study changes expert class scores, not its gate.
No claim that shrinkage covariance, RBF kernels or residual learning are new.

## One module, isolated components

Use the same192 cached source-PCA visual coordinates and source-only
centering/scaling as the archived linear residual expert. Let x be those
coordinates divided bysqrt192. On outer-source rows ONLY, subtract the
source mean of each true class and fit sklearn LedoitWolf(assume_centered=True)
to the resulting within-class residuals. All source rows contribute; covariance
estimation is row-weighted, NOT class/port balanced. Form its symmetric inverse
square root W. Normalize xW so its average SOURCE squared norm equals that
of x. This seeks to suppress source-observed within-class variation, but does
not establish that such variation is nuisance or transferable to new ports.

Choose1024 source rows without replacement with RNG seed1707 (or the largest
even number available if fewer). The median squared distance of512 disjoint
pairs sets RBF denominator h separately in each geometry. Kernel:
K(x,z)=1+exp(-||x-z||^2/h). No target samples or labels determine W or h.
The constant1 keeps the same penalized global-bias component as the old head.
Far-away support influence decays only for the local term, not the globalbias.

Three fixed new arms:
- metric_linear: kernel1+<xW,zW>, testing source geometry alone.
- isotropic_rbf: kernel1+exp(-||x-z||^2/h), testing locality alone.
- metric_rbf: both changes; primary candidate.

All use sklearn KernelRidge(alpha=8,kernel='precomputed') on exactly the
original64 target support rows. Responses are2*one_hot(y)-1-source_scores.
Support weights are64/(number_of_observed_classes*class_count), as before.
Prediction adds kernel residual to the unchanged source head. No new
backbone/PCA/source classifier fitting, new target labels or parameter search.
The old linear and bias experts are unchanged controls, not refit candidates.

## Deployment and readout

23 ports, original five supports each. Exclude those64 rows from evaluation.
Reuse original source-trained4D confidence gate parameters, recomputing its
inputs on each actual receiving expert. Require positive gate score, actual
base/expert disagreement and a support-observed expert class. At5% primary
admission cap floor(.05*N_eval), rank with global-row tie breaks. Existing
frozen bias/visual pipelines remain exact controls. Also retain the previous
action-advantage visual policy as a whole-policy comparator (not as a second
new gate or a criterion for picking a gate). No other budgets are evaluated.

Primary development screen: metric_rbf mean5% BA > archived linear visual.
Separately report metric_rbf minus isotropic_rbf and minus metric_linear to
assess its two components. Do not claim both components help unless their
matched ablations support that claim. Report all arms even if negative.

Full-query expert BA/accuracy are secondary diagnostics, not5% results.
Also report rescues of cases where BOTH base and archived linear visual
expert are wrong, and harms of formerly correct linear predictions. These
diagnostics use query labels AFTER all policies have been frozen and never
change candidate training, masks or selection. Capture full class recalls.

Save geometry, source bandwidth pairs, dual coefficients, expert scores,
gate scores, masks and predictions for ALL ports before any candidate query
readout. Evaluate equal-port metrics over five supports and descriptive
paired port-bootstrap intervals; five draws are not independent datasets.
Stop after this fixed screen, with no post-result kernel/strength sweep.

## Limits

All targets have historical development exposure. Source head/PCA purges are
inherited from archived producer checks; metric fitting is outer-source only.
Support/query isolation is ROW ONLY, not vessel/product-disjoint. The labels
used for source covariance are source labels, separate from the64 target
label budget. All query receiver scores precede selection, so this is an
admission cap, not measured compute savings. The module is a testable method
combination, not established top-journal novelty or guaranteed positive gain.
