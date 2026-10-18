# Source-traceable integration pilot

Frozen 2026-10-18 before new classifier outcomes. Reuse only the previous seven
raw classes; no class merging or gain-based cohort selection. Index is restored
at E:/Hermes/paper_tracking/f0921_chips_clean_npy/index.csv. Exact VV/VH replay
against current final SHP/TIFF establishes pixel lineage, not independent AIS
truth. Keep unique replay links, detector-sized crops, and consistent identities.
The five date/UTM groups come from the already chosen source products; all exact
linked observations in those groups can enter, not only the original five codes.
No relabeling or modification of datasets. This remains a bounded clean-crop
development cohort, not a full-scene test or independent confirmation.

Hold out an entire observation date and purge its MMSIs from supervised source
fitting. Within a date retain one deterministic observation per ship; reserve
max(8, ceil(0.3*n)) query ships using label-blind seed20261018. The remaining
ships form the target support pool. Evaluate nested B16/32/64 if capacity allows,
five random draws0..4, with the same queries and supports across every arm.
No forced target class coverage. Skip dates unable to supply B16 and8 queries;
report them. Source fitting uses one observation per source ship and requires
all seven source classes. Query labels never select weights, support members,
checkpoint or readout. This is few-shot held-date recognition; supports can be
from the same acquisition as queries and must not be called source-only support.

Frozen ImageNet R50 sees paired grayscale channels independently at128. Frozen
24-port-only SSL R18 sees112 RGB=[VV,VH,mean]. Audit its existing training AND
selection records against this cohort identities and acquisition keys. This is
external SAR pretraining, not F-specific source-only SSL. No backbone retraining.
Re-extract all features on these new inputs; never join old feature rows.

Refit separate seven-class source ridge priors on R50/R18 global features.
Source-centered, L2-normalized global vectors and class-balanced strength1 ridge
use the existing unified_head_core coefficient solver. Support adaptation uses
its strength8 class-balanced residual solve; update SS and QS together. Average
the two adapted heads' softmax probabilities at fixed0.5/0.5. R18 and both source
priors are identical across readout arms. No temperature/fusion/weight sweep.

R50 kernels: global; global_grid=.5global+.5grid2; grid12/postpool/prepool/
target_only=.5global+.25grid2+.25new. Last four exactly retain the previous
readout definitions and centering/normalization. Target_only substitutes prepool
target slots into postpool while preserving other slots. Include a strength
control=.5global+.25grid2 to expose old-kernel attenuation. Class-reference/C4
learned readers from the old eight-class system are NOT silently reinstated.
This validates the full prior-support-fusion decision path, not every historical
module. Compare final system, individual heads and retrieval separately.

Obtain masks with frozen550k detector, conf.25/iou.3, each stored224 channel padded
to1024 without rescaling. Target is the unique box containing the documented
crop anchor per polarization; if ambiguous, that polarization supplies no target
box. Union valid target boxes and all other candidates; subtract target from
other, remainder is background. No AIS angle or class input. Retain all enrolled
queries when masks missing; report availability, no outcome-based filtering.
These are candidate masks, not verified physical components or detection truth.

Report equal-date and query-weighted OA/BA, NLL/Brier, MAP@R, paired draw spread,
class/support composition, rescue/harm and no-target strata. All repeated draws
reuse queries, not independent samples. Review original stored224 VV/VH plus
actual same/confused supports and source-native windows for explanatory cases.
No physical mechanism naming from appearances. Archive scoped code/results,
not original images, weights or feature arrays, then verify GitHub push.

Post-run provenance clarification (does not alter the frozen evaluation): the
current source TIFFs are daily first-nonzero mosaics with overlap. Pixel replay
establishes crop-to-mosaic lineage, not per-pixel single-product ownership.
Eighty-nine enrolled records lack a recovered full product ID; retain unknown.
Whole-date exclusion prevents same-date mixture crossing supervised source and
target partitions. Record-level product matches do not establish pixel ownership.
The two B32 target-slot rescues have no other-candidate coverage; do not attribute
them to neighbor propagation. Strict cross-acquisition support retrieval remains
a separate experiment, not a claim of this few-shot held-date pilot.
