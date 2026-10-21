# Crude tanker input controls, 2026-10-21

## Questions fixed before fitting

1. Is the observed PNG contrast association affected by unknown display processing?
2. Does outside-OBB content help classification when each input has its own source-trained head?
3. Can an apparent body readout still use outside content through convolution?
4. Which observation conditions are actually represented by the source training ships?

This is a mechanism diagnostic on the existing 24-port corpus, not a new independent confirmation dataset. The 100 crude tanker pairs were identified in previous analysis. They are not a new test set. No target support labels or model selection are used.

## Matched factorial comparison

Frozen ImageNet R50 stem; native 160-pixel crop/pad, no resizing; VV/VH processed separately; exact four quarter-turn readouts averaged. Pixel values remain the existing PNG values. A valid target OBB is required in VV. VH is zeroed in every arm when its target mask is missing or the recorded cross-polarization association is disabled. Observation 31237 remains excluded.

| Arm | Input before convolution | Pooling footprint |
| --- | --- | --- |
| full_target | Full native chip | Target OBB |
| body_target | Outside target set to zero | Target OBB |
| full_global | Full native chip | Valid original chip pixels |
| body_global | Outside target set to zero | Valid original chip pixels |
| outside_global | Target set to zero | Valid original chip pixels |

All arms have the same 256 mean/max response dimensions plus four OBB pixel length/width values. No port, absolute angle, incidence or neighbor class is fed into the classifier. Outside-only is a diagnostic of available context/shape association, not a deployable target-free ship classifier. The hard mask and its boundary are visible; these are not interventions that reproduce natural ocean conditions.

The two pooling footprints are crossed with the input restriction, so different pooling is not silently mistaken for the effect of background. In particular full_target is NOT strict body-only: its receptive fields can include outside pixels. Both inputs are masked before normalization/convolution. Padding is excluded from pooling.

## Fit and evaluate

Reuse the three existing identity folds and calendar train/validation/test boundaries. Fit each arm separately with identical source rows, class/ship weights and source-only standardization. Ridge alpha in {0.1,1,10}, selected by source validation BA then Brier. These are deterministic fitted classifier heads, not three random neural-training seeds or end-to-end backbone training. All eight classes stay in training.

Report OA, BA, macro-F1, each class recall, Brier and NLL (softmax T=1, not claimed calibrated). For crude pairs report both/one/neither correct and contrast-margin association. Paired confidence intervals resample held-out MMSI, conditional on fitted heads; they do not capture model-training or independent-acquisition uncertainty.

Primary comparisons: full_global vs body_global, full_target vs body_target, and body_target vs full_global as the natural two-pipeline comparison. All are reported, including negative results. No arm or query is chosen according to target performance.

## Image perturbation diagnostic

After fitting, apply fixed +/-0.05 PNG-unit shifts separately to target pixels and to rest pixels (excluding other detected objects), separately for VV and VH, retaining geometry, target scale and all other pixels. Clip to [0,1] and record clipped fractions. Use the full_global frozen head. This tests this new model's response to controlled displayed-intensity changes, NOT physical backscatter changes or the old model's causal mechanism. Do not retrain or select shifts on test outcomes.

## Literature reference

Awais et al., Cross-Dataset Transfer and Unknown-Class Detection in Imbalanced SAR Ship Classification, arXiv:2609.31658v1. Read local full text, especially Sections 2.3, 2.5, 3, Table 5, supplementary Tables 7-10. Adopt BA/macro-F1/per-class reporting. No unknown-class AUROC is claimed here: all eight classes are known. Their cross-dataset three-class numbers are not directly comparable with this eight-class identity/time split. Their error-detection and unknown-detection labels are distinct and must remain separate in our implementation.

## Completion criteria

Publish actual data provenance limits, source condition coverage, all matched results and representative raw VV/VH cases. Preserve old findings. No claim that crude tankers are physically weak scatterers, no claim that unsupported conditions are absent because a histogram bin is empty, and no gain claim from a correlation alone.
