# Complementary expert: new rescues do not outweigh new harms

## Decision

The prespecified cross-fitted complementary expert fails its primary5%
comparison. Keep the current50.382098% BA visual-adapted-base handoff as the
main baseline. Neither this study nor its separately declared error-focused
follow-up establishes a new positive module. Do not replace the manuscript's
main result or substitute per-port winners.

Output: `E:/Hermes/paper_tracking/sensitivity/oof_complementary_expert_results_20261007`.
Code: `complementary_expert.py`, `run_complementary_expert.py`.
Protocol: `COMPLEMENTARY_EXPERT_PROTOCOL_20261007.md`.
23 ports, five original64-label supports,345 new expert fits,690 policy rows
and460 full-query expert rows,11.00 seconds. Reused7,360 verified base LOO
fits, no alpha selection and no new base/source-head/backbone/gate fitting.

## Tested Mechanism

The unchanged B64 base remains the deployment fallback. Expert fitting uses
the residual to Bminus_i + (E0-B0), where Bminus_i is each support row's
63-label held-out base score. Deployment adds its fitted correction to
B64 + (E0-B0). The explicit source expert/base difference retains existing
source-learned information; it is not new facility/AIS/geography knowledge.

Controls replace held-out base scores with in-sample B64, or remove E0-B0
from BOTH training and deployment. Features,64 labels,lambda8 and class
weights match. The final expert is trained on all64 labels, not independently
validated on its own support. The frozen gate is applied to actual receivers.

## Budgeted Results (%)

| Policy | BA | Accuracy | Expert admissions |
| --- | ---: | ---: | ---: |
| Adapted base alone | 48.540070 | 54.794330 | 0 |
| Current visual handoff | 50.382098 | 56.653819 | 4.932516 |
| Current bias handoff | 49.929602 | 52.635672 | 4.932516 |
| Cross-fitted complementary expert | 50.250454 | 56.447164 | 4.932516 |
| In-sample complementary expert | 50.205085 | 57.097913 | 4.932516 |
| Without explicit source score contrast | 49.329688 | 55.366558 | 4.771164 |

Primary difference is -0.131644 BA points:9 positive/14 negative ports,
descriptive port-bootstrap95% interval[-0.305574,0.022989], four of five
draw means negative. Draw differences:
[0.104915,-0.078413,-0.281334,-0.211456,-0.191930]. Accuracy -0.206655 points.

Crossfit minus in-sample is only+0.045369 BA points,10 positive/13 negative,
interval[-0.134386,0.255862], three draw means negative. Cross-fitting is
not an independently established beneficial component here.

Crossfit still exceeds adapted base alone by1.710384 points,20/23 positive,
but the current expert already adds1.842028 points. Beating base alone is
not sufficient to claim the new expert improves the existing workflow.

## Full-Query Expert Diagnostics (%)

These are NOT5% deployment results. Recovery/harm references the adapted base.

| Expert | BA | Accuracy | Recovers vs base (BA points) | Harms vs base (BA points) |
| --- | ---: | ---: | ---: | ---: |
| Current anchored expert | 52.894867 | 58.425765 | 9.273766 | 4.918970 |
| Crossfit complement | 52.822839 | 58.214313 | 9.323835 | 5.041066 |
| In-sample complement | 52.768036 | 60.739806 | 9.043123 | 4.815157 |
| No explicit source contrast | 49.849166 | 55.553625 | 4.178113 | 2.869017 |

Crossfit rescues1.772173 BA points on rows where BOTH adapted base and old
expert were wrong. However, across all query rows it recovers2.902909 points
from old-expert errors while losing2.974936 points on old-expert-correct
rows. Net -0.072027 points. Genuine complementary corrections exist, but
their harm/recovery tradeoff is negative, not merely a ranking issue.
Full-query Mombasa declines6.023150 points relative to the old expert.

## Source-Information Ablation

Retaining E0-B0 versus removing it gives+0.920766 BA points at the5% cap,
21/23 ports positive and all five draw means positive. Actual admissions
differ, so do not call this an equal-actual-count comparison. The separate
full-query comparison also favors retention by2.973673 points,21/23 positive,
interval[2.105366,3.845187]. Thus the existing learned source score difference
is useful IN THIS construction. It does not prove a new knowledge module,
pure VH attribution, or superiority over the original expert which already
contains that source information. Removing D does not remove all source
knowledge: source features and B0 remain.

## Follow-Up And Limits

After this negative readout, a separate fixed error-weighted expert was
declared before its own readout. See `ERROR_FOCUSED_PROTOCOL_20261007.md`
and `ERROR_FOCUSED_FINDINGS_20261007.md`. It preserves E0 rather than this
new score decomposition; it is not an original preregistered arm or rescue
of this failed primary. Both negative studies remain visible.

All targets are development-exposed, support/query separation is row-only,
and all expert scores are accessed before selection.5% caps handoffs, not
changes from original source predictions or measured compute. Retained
crossfit policy changes30.639102% of original base predictions. The unchanged
gate is not newly calibrated for these receiver score distributions.

## Verification

Four focused tests passed: independent augmented-SVD fit, reduction to the
old anchored expert when the base is not adapted, deployment/training score
separation and exact zero residual. All three baseline policies and old full
expert metrics reproduce in the producer. Independent replay passed in7.45s:
345 augmented-SVD target correction refits,345 training priors/score replays,
575 gates/masks,690 policy predictions,690 policy plus460 full-expert metric
rows/class recalls, and all ten summary contrasts.345 exact baseline policy
controls and115 old full-expert score arrays are unchanged. The no-source
control removes its explicit contrast from both training and deployment.

Eight independent63-row base-LOO SVD refits at the first two sorted ports,
two draws and rows0/63 confirm cache usage. Other LOO entries inherit prior
verification; this is not a new exhaustive7,360-fit LOO audit. Maximum
coefficient error1.05e-14, score1.62e-14, LOO2.00e-15, gate4.60e-14 and
metric3.33e-16. See the output's `VERIFICATION.json`. No tolerance changes
were needed, and artifact replay does not establish independent validation
or retrospectively certify the freeze timeline.
