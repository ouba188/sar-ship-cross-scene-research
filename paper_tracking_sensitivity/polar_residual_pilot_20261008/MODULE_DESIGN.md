# Competitor-Directed Polarization Residual

## The Information Change

Previously the extra expert encoder saw VH only and reduced its spatial map
to global average pooling. The base R50 already consumed VV/VH, so this is
not a claim of adding a previously unavailable modality to the entire system.
The new expert receives paired VV/VH spatial maps and learns which spatial
polarization differences support its competing class hypotheses.

Both bands are contrast-stretched image chips. Differences below are learned
feature differences, NOT calibrated physical backscatter ratios or phase data.
No port facilities, AIS trajectories or geographic metadata enter this module.

## Forward Definition

Let h_s and v_s denote shared-R18 layer4 VH/VV features at spatial location s.
Let m=mean_s(h_s), d_s=v_s-h_s and c_s=(v_s+h_s)/2. With128x128 inputs the
layer4 grid is4x4, so the module keeps16tokens instead of pooling immediately.

The retained source expert supplies its two highest prior classes, a and b,
on the unfused VH feature m. Its exact class-difference direction in the
512-dimensional extra-feature space is:

    u = ((W_b,SSL - W_a,SSL) / scale_SSL) @ PCA_SSL

W is the retained ridge classifier. This is the derivative of its b-minus-a
prior score with respect to the extra feature, not a separate learned class
prototype and not a target-query label. The direction is normalized before
projection into a64-dimensional query. Source-port episodes use their own
source-excluded prior; deployment uses the fixed global source prior.

Layer-normalized common/difference tokens supply keys and a shared content
score. The query-key interaction supplies a class-competition score p_s:

    a_plus  = softmax(content + p)
    a_minus = softmax(content - p)
    contrast = sum_s((a_plus_s - a_minus_s) * d_s)
    descriptor = mean_s(d_s) + contrast
    expert_feature = m + Linear512to512(descriptor)

The last linear projection has no bias and is zero-initialized. Initial
features therefore reproduce VH pooling; learned updates can add global and
competition-conditioned spatial VV-VH evidence. Identical paired maps yield
zero residual even after training. A spatially constant d yields no contrast.
These are structural properties, not proof of accuracy improvement.

The new feature enters the SAME fixed PCA/normalization/source prior and
64-support ridge adaptation during training and deployment. Training labels
supervise the adapted source query prediction; gradients traverse support
and query features. No auxiliary classifier is discarded after training.

## Bounded Test

The protocol is frozen in protocol.json before target readout. Two6epoch
training arms share source episodes, initialization and optimization schedule:

- polar_pooled: learn the global VV-VH residual only.
- polar_contrast: also learn the competitor-directed spatial contrast.

contrast_no_spatial uses the SAME trained contrast model with only its
contrast term disabled at inference, retaining the global polarization
residual. It measures that term's effect within the trained model and is
not an independently trained control. The prior aligned-global VH model is
also retained for direct pipeline comparisons. All use identical target64
supports, query rows, frozen base predictions and5%handoff constraints.

The optimizer uses1e-5forlayer4 and1e-3fornewmoduleparameters, weight decay1e-4,
clip5. Parameters controlling the retained head/gate are not refitted. Final
epoch only; neither early stopping nor target-based parameter search occurs.
New modules have additional parameters and paired encoding costs more than
VH alone. Equal epochs/rows do not mean equal FLOPs.

64evenly spaced source and64target prefix pairs were read before training:
none were identical. Mean absolute VV-VH prefix difference was0.330239 and
0.378605respectively. This checks that the expert actually has a distinct
input branch, not that all128pairs contain useful ship evidence.

## Prior Art And Claim Boundary

Shao, Zhang and Ke, A Dual-Polarization Information-Guided Network for SAR
Ship Classification, Remote Sensing15(8):2138,2023, doi:10.3390/rs15082138,
already uses polarization cross-attention and feature fusion. Publisher-
deposited Crossref metadata confirms the title/authors and PCCAF/DRDLF in
the abstract. Do not claim dual-polarization attention itself as new.

The hypothesis tested here is more specific: condition paired spatial
contrast on the retained receiving expert's actual class-difference geometry,
and learn it through the same limited-support residual adaptation used at
deployment. This local formulation has NOT received an exhaustive novelty
search. Accuracy gains, component attribution and broad novelty remain
separate questions. Results belong in POLAR_FINDINGS.md after readout.
