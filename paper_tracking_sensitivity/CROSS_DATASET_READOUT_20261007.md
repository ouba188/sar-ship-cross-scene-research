# Fixed current-method cross-dataset readout, 2026-10-07

All values below are percentages or percentage-point differences, averaged
over the same five support seeds0..4 within each dataset. Every draw uses64
target labels shared by both adapted branches and at most5%expert admissions.
No target hyperparameter search was performed in these runs.

| Target protocol | Target rows/classes | Original BA | Adapted-base BA | Full BA | Handoff increment | Positive draws |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| OpenSARShip1 native GRDH fine-label subset, paired input | 1349 /8 | 21.233037 | 31.510334 | 32.330577 | +0.820242 | 5/5 |
| OpenSARShip2 explicit fine-label subset, paired input | 3745 /8 | 19.457723 | 29.295273 | 29.938364 | +0.643091 | 5/5 |
| FUSAR fine-label subset, single-image input adapter | 1158 /7 | 22.463793 | 28.446428 | 28.894663 | +0.448235 | 4/5 |
| F_0921 existing clean fine-label subset, paired input | 123489 /8 | 18.846647 | 20.441980 | 20.839997 | +0.398017 | 4/5 |

The full model's incremental handoff benefit is positive ON AVERAGE in all
four fixed protocols. It is small; FUSAR has a negative draw(-0.685866pp)
and F0921 has a slight negative draw(-0.004176pp).
Do not present the much larger base-adaptation gains as expert-handoff gains,
average these BA values as a common benchmark, or treat twenty draws as twenty
independent datasets. No statistical significance claim is established here.

Source taxonomy and model settings are unchanged. Only compatible target
fine-label rows are included. No row is performance on the complete
standard dataset taxonomy. Full source fitting cohorts differ because the
existing target identity exclusions differ.

FUSAR duplicates one image into the paired encoder's two channels and uses
the same image in the SSL encoder. It does not provide a second polarization
or validate polarization complementarity. Target support/query is row-level,
not vessel-disjoint; all four datasets have historical development exposure.
F0921 additionally supplied the e206 SSL pretraining corpus, so its result is
known-corpus/transductive-encoder transfer, not independent external evidence.
Its label budget is64GLOBAL labels despite the much larger target population.
Its source purge leaves23ports, including singleton Houston/NewYork groups
under the unchanged equal-port gate weighting; see the detailed limitation.
The5%constraint limits handoff counts, not computing all expert features.

Detailed artifacts:

- opensarship1_transfer_20261007/TRANSFER_FINDINGS.md
- opensarship2_transfer_20261007/TRANSFER_FINDINGS.md
- fusar_transfer_20261007/TRANSFER_FINDINGS.md
- f0921_transfer_20261007/TRANSFER_FINDINGS.md

These are current-model transfer studies, distinct from the older
external_os2_fusar within-corpus experiments using different base features,
taxonomies, gates and budgets. Old caches and results remain preserved.

## Ordinary Accuracy And Matched F0921 Input Correction

OA is ordinary accuracy; values below are percentages, not BA.

| Target protocol | Source-only OA | Adapted-base OA | Full OA |
| --- | ---: | ---: | ---: |
| OpenSARShip1 native GRDH fine-label subset | 38.163424 | 40.513619 | 43.066148 |
| OpenSARShip2, preceding fixed protocol | 18.565607 | 30.638413 | 32.382505 |
| FUSAR, preceding fixed protocol | 46.965265 | 51.608775 | 52.650823 |
| F0921, original cached224 inputs | 17.920356 | 21.433259 | 21.834150 |
| F0921, matched detector recrop | 24.207575 | 28.937087 | 29.633543 |

Matched recrop is a same-dataset input-pipeline comparison, NOT a fourth
independent dataset. It retains123489 rows, original global64 support draws,
source model and5%handoff caps.76568rows use DET-centered source-sized crops;
46921unresolved/no-detection rows retain exactly their original input/features.
The old cross-tree positional join was unsafe; the new join uses scene+MMSI
with explicitly recorded ambiguity handling. Source radiometry remains unresolved.

Recrop BA original/base64/full5 =20.314242/24.901788/25.105149%.
Full-model pipeline correction gives+7.799392ppOA /+4.265153ppBA, positive
in all five matched draws. Within the corrected pipeline, handoff adds only
+0.696455ppOA /+0.203362ppBA, withBApositive3/5. Do not claim the pipeline
gain as a new module contribution. Detailed evidence and limitations:
`f0921_detector_recrop_20261007/RECROP_FINDINGS.md`.

## Generic Cross-Domain Mechanism Validation

Separate from SAR cross-dataset transfer, the same adaptation/handoff rule was
run on all four PACS domains(7classes) and OfficeHome domains(65classes).
Each target-domain task uses64labels and5%query handoff. Equal-domain means:

| Dataset | Base64 OA | Full OA | Base64 BA | Full BA | Handoff BA pp | Positive domains |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PACS | 76.024350 | 77.969497 | 78.237061 | 79.868998 | +1.631937 | 3/4 |
| OfficeHome | 66.558133 | 67.285482 | 64.852800 | 65.530986 | +0.678186 | 3/4 |

Keep negative domains: PACS/photo-3.213599ppBA and OfficeHome/Art-0.090568ppBA.
These use two encoders of the same RGB image; they do not validate SAR
polarization or port knowledge. Existing source-supervised B encoders exclude
the outer target according to extraction code but saw inner held-source labels;
source OOF is head-level conditional on fixed representations, not encoder-OOF.
Historical data, no blind/SOTA claim. Full domain readout and budget scope:
GENERIC_CROSS_DOMAIN_READOUT_20261007.md.

## Bounded F0921 Representation Pilot

Separate development experiment: neural_complement_pilot_20261007/PILOT_FINDINGS.md.
Same corrected inputs,64labels,5%handoff and fixed R50 base; only expert R18
layer4 is trained on source labels. Full OA/BA: frozen replay29.634353/25.106684,
ordinary CE29.247559/24.997577, utility-tail29.565971/25.319728.
Candidate beats ordinary CE by+.318412ppOA/+.322151ppBA(all5draws), but versus
frozen replay gives-.068382ppOA/+.213044ppBA. No main-model replacement or
new independent dataset. Training-only utility head is not an attention module.

Successor: aligned_residual_pilot_20261008/ALIGNED_FINDINGS.md fixes the
discarded-classifier mismatch by differentiating through the actual64support
ridge expert and retaining source PCA/prior/gate at deployment. Aligned-global
FullOA29.326474/BA25.435555: vsfrozen-.307879ppOA/+.328871ppBA.
Retained asusableBA-orientedcandidate peruserpreference, withtradeoffreported.
ExtraadmissionCE lowersbothmeans versusmatchedaligned-global; notenabled.
See CANDIDATE_MODULES.md forobservedpositivevariants andtheirusable scope.

Nextinformation-levelprototypes actuallyaddVVspatialmaps totheVHexpert:
polar_residual_pilot_20261008/POLAR_FINDINGS.md and
support_visible_polar_20261008/SUPPORT_VISIBLE_FINDINGS.md. Bothunconstrained
and source-prior-preserving spatialcontrast failtomatch pairedpooling or
aligned-globalBA25.435555. Donotclaimtheirpositiveold-referencecomparisons
asattentiongain. Newinformationpathimplemented; usefulcoremoduleunestablished.

## Information-Loss Diagnosis (2026-10-08)

Completed CPU probes include24ports, correctedF0921, publicOSS1 andOSS2.
Read information_loss_diagnosis_20261008/INFORMATION_LOSS_FINDINGS.md.
Same-vessel-separated diagnostic64/512/4096-label probes are NOT production
64-label transfer results. At512labels, raw joint features improve bothOA/BA
over PCA joint on allthree external datasets, but not universally on24ports.
At64labels OSS2 raw-jointBA slightly declines; no general methodgain established.
F0921 input-only sourcehead gain+6.287219ppOA, withzero fallbackpredictionflips,
locates a real input-construction contribution; classwise recovery is uneven.
Large24port geometry gains are confounded: fields present only forclasses0/1/2,
missing forclasses3..7. Missingness alone reproduces most apparent improvement.
Complete-case3class geometryBA is nearly unchanged. Do not promote the geometry
arm as a scale-aware module. Nextpriority: test preserved rawfeature information
with the existing64label/prior/handoff setup before anotherattention module.

## Targeted Complement Expert Pilot (2026-10-08)

The above64label test isnowcompleted onF0921/publicOSS1/publicOSS2; see
projection_residual_pilot_20261008/PROJECTION_FINDINGS.md. FixedhalfPCA/half
sourcePCA-complement kernel changes onlysupport expertadaptation, preserving
sourceprior, Base64, gate and5%handoff. FullOA changes+.195260/+.217899/+.059766pp;
BA changes-.101876/-.359902/+.033502pp respectively. ExpertOA improvesinall5draws
oneachdataset. RetainasanOA-orientedcandidate; donotclaimuniformBA improvement.
Simplehalf-retained control improvesF0921BA+.198438pp butisnot newinformation.
Source-supervised8Dcomplement follow-up degradesbothOA/BA onallthree andisdisabled;
see source_discriminant_pilot_20261008/SOURCE_DISCRIMINANT_FINDINGS.md.
No24port evaluation ofthismodule ornewblindconfirmationhasbeencompleted.

## Joint Four-Module F0921 Pilot (2026-10-08)

Matched2x2 integration completed: joint_residual_pilot_20261008/JOINT_FINDINGS.md.
The final complement kernel is inside the sourceepisodic traininggradient;
thisisnot a posttrainingbranch attachment. Allarms share64labels, exactBase64,
fixedgate and6171/123425handoffs. FullOA/BA: reference29.634353/25.106684,
complement29.829289/25.003473, aligned29.326474/25.434966,
joint29.758639/25.431755. Jointvsreference+.124286ppOA/+.325071ppBA.
Vs aligned-only, OA+.432165pp(all5draws), BA-.003211pp; vscomplement,
BA+.428282pp(all5draws), OA-.070650pp. RetainjointasbalancedF0921candidate.
Thesearematchednewround values, notsplicedfromslightlydifferentoldreplays.
JointpublicOSS/24port results donotexistyet; priortables remainunchanged.
