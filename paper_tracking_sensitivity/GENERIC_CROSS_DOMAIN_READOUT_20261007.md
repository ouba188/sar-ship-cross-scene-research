# Same64 /5% cross-domain mechanism validation, 2026-10-07

## Aggregate Result

Percentages; equal-domain averages of five support-draw means.
All original classes and all four target domains are retained.

| Dataset | Classes | Original OA | Base64 OA | Full OA | Original BA | Base64 BA | Full BA | Handoff OA pp | Handoff BA pp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PACS | 7 | 67.529691 | 76.024350 | 77.969497 | 67.348009 | 78.237061 | 79.868998 | +1.945148 | +1.631937 |
| Office-Home | 65 | 65.520207 | 66.558133 | 67.285482 | 64.120288 | 64.852800 | 65.530986 | +0.727349 | +0.678186 |

Both dataset means improve, but only3/4domains improve in each dataset.
Do not claim universal improvement, SAR encoder generalization, SOTA, or
independent statistical significance from these overlapping support draws.

## Every Domain

| Dataset/target | Query rows per draw | Base64 OA | Full OA | Base64 BA | Full BA | Handoff BA pp | BA-positive draws |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PACS/art_painting | 1984 | 78.6089 | 80.6552 | 78.3956 | 80.6124 | +2.2168 | 5/5 |
| PACS/cartoon | 2280 | 70.7895 | 74.7456 | 74.4362 | 77.8644 | +3.4282 | 5/5 |
| PACS/photo | 1606 | 97.7210 | 95.1059 | 97.1991 | 93.9855 | -3.2136 | 0/5 |
| PACS/sketch | 3865 | 56.9780 | 61.3713 | 62.9174 | 67.0138 | +4.0963 | 5/5 |
| OfficeHome/Art | 2363 | 66.5595 | 66.3479 | 62.9545 | 62.8640 | -0.0906 | 2/5 |
| OfficeHome/Clipart | 4301 | 49.4769 | 51.3648 | 49.6305 | 51.3427 | +1.7122 | 5/5 |
| OfficeHome/Product | 4375 | 74.2400 | 74.8069 | 72.4389 | 72.8935 | +0.4546 | 4/5 |
| OfficeHome/Real World | 4293 | 75.9562 | 76.6224 | 74.3872 | 75.0237 | +0.6365 | 5/5 |

PACS/photo is a substantive failure case. The adapted base already achieves
97.72%OA. All five policies admit80rows: rescues9/14/15/13/19 versus
harms60/59/54/58/49. The positive source gate does not prevent harmful handoff
under this target shift. Although5%is a maximum rather than forced admission,
enough positive-scored candidates fill the cap. No target-based threshold
repair, domain removal or budget retuning was performed after this readout.

## Protocol And Representation

Each dataset uses four outer leave-one-domain-out tasks. Each outer target
domain has its OWN64label budget per draw, shared by both adapted branches.
Thus one seed across the four tasks uses4x64support annotations, not64total
across the entire four-domain benchmark. All remaining rows are query only.
Five fixed support seeds0..4; no class quotas or frequency-based class removal.

Within an outer task, fit PCA128base/PCA64extra, source-balanced ridge1heads,
and a source-domain/class-balanced4Dconfidence LR gate with C1. Source head-OOF
uses three inner held-source-domain folds. Final heads use all three source
domains. Adapt both branches with residual ridge8 on the same64target labels.
Admit only query disagreements whose expert class was observed in support and
whose frozen gate score is positive, ranked deterministically by gate score
then row ID, up to floor(.05*N_query). Preserve all class output columns.
All five policies are saved before query metric computation.

Admission counts in every draw: PACS99/114/80/193; OfficeHome118/215/218/214
in the domain order above. PACS support covers all7classes in every draw.
OfficeHome64labels cover34-46of65classes depending on domain/draw; other
classes are not dropped from query evaluation and cannot receive expert handoff.
The65class setting therefore differs materially from the old top8 experiment.

Existing local data/feature provenance:

- PACS9991rows; A(9991,2048), B_target(9991,512), four outer-fold B caches.
- OfficeHome15588rows; A(15588,2048), B_target(15588,512), four B caches.
- A is frozen ImageNet-V1 ResNet50. B is ImageNet-V1 ResNet18, fine-tuned
  for12epochs with labels from the three outer source domains, per existing
  extraction code. Both encode the SAME RGB image, not separate sensors.
- Caches follow original parquet order; OfficeHome concatenates sorted shards.
  The NPY arrays have no embedded sample IDs. Alignment is supported by code
  and dimensions, not independently authenticated identities. Encoder training
  seeds were not recorded in the old producers.

This validates the adaptation/routing mechanism with dataset-specific visual
representations. It does NOT apply the frozen SAR8class classifier to generic
objects, test port knowledge, or validate physical-polarization complementarity.

## Encoder Exposure And Old-Result Differences

According to extraction scripts, B_target excludes the named outer target's
labels. However B_target trained on ALL THREE source domains, including the
domain held out in an inner head-OOF fold. Consequently OOF independence applies
to PCA/scalers/heads conditional on this fixed representation, NOT encoder
training. This is not a strictly nested encoder-OOF study. Replacing B_target
with B_inner would expose outer-target labels and is not a valid shortcut.

Both datasets had historical development use on2026-10-05; these are not new
blind-confirmation data. Old pair-key experiments accessed target truth;
old full-chain readouts included64support rows in target metrics, and the
OfficeHome full-chain selected8classes using global frequencies. Those results
are not reused as current-protocol evidence. The new pipeline uses none of
those selection/readout procedures and preserves the old artifacts unchanged.

## Implementation And Verification

New loader/tests: oof_reliability_check/generic_transfer_data.py and
test_generic_transfer_data.py. All eight real outer folds load their original
class mappings; eight loader tests pass.
New runner/tests: run_generic_transfer.py and test_generic_transfer_model.py.
Four model tests verify7/8/65class residuals against a weighted primal Ridge,
exact8class gate parameter replay,65class support/5%constraints, and metrics
against sklearn. Four shared transfer tests and one batched projection test
also pass. Total17tests in these generic integration invocations.

Only two shared calculations changed: residual targets infer class count from
the prior columns, and the observed-class guard uses the same count. Existing
eight-class behavior is retained. No dependency or encoder training was added.

Outputs: pacs_handoff_transfer_20261007/ and
officehome_handoff_transfer_20261007/, each containing four domain directories
with protocol, source model, source-head OOF scores, frozen policies, query-only
metrics and summary; each dataset root contains equal-domain summary.json.
