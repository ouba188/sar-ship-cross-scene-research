# Two-paper sidecar: 2026-10-13

Scope: primary methods/experiments only; no experiments or additional papers. Page references below identify the accessed versions, not unverified final pagination.

## DPIG-Net

Publication: Remote Sensing 2023, 15(8), 2138; published 18 April 2023; DOI [10.3390/rs15082138](https://doi.org/10.3390/rs15082138). [Publisher](https://www.mdpi.com/2072-4292/15/8/2138) metadata was searchable, but full-page/PDF requests returned errors. Fulltext read: [arXiv:2207.04639v1](https://arxiv.org/pdf/2207.04639), 11 July 2022, Tianwen Zhang and Xiaoling Zhang, five pages. Do not silently attribute preprint numbers to the final journal version.

- Inputs, Sec. II, p.2: THREE branches: I1=|SVH|, main I2=|SVV|, I3=|SVV*conj(SVH)|. VV's stronger ship scattering motivates its privileged role. Auxiliary polarization features are intended to recover information omitted by VV-only classification.
- Fusion, Sec. II-A/B, pp.2-3: four-stage CNN encoders; VV-conditioned cross-attention on auxiliaries; concatenate; three dilated residual dense blocks; global VV residual protects main-branch details; FC/softmax. Sec. III-B: supervised CE, from scratch, Adam, 100 epochs.
- Evidence, Sec. V-A, Table VII, p.4: six-class VV-only 52.69%, two-input 54.16%, three-input/no-cross-attention 56.83%, full 58.68%. Table VIII favors VV as main. These support configuration utility, not isolated physical causality. Sec. IV-A averages best 10 of 20 trainings.
- Evaluation: OpenSARShip SLC three-/six-class internal splits, not cross-dataset transfer. Authors explicitly require paired complex VV/VH; no AIS inference branch. A single uint8 chip cannot supply paired polarizations.

Independent mathematical caution: as printed, |a*conj(b)|=|a||b|. Thus this equation alone does not establish phase dependence or a necessity for complex values once paired amplitudes exist. Preserve the distinction between the authors' complex-data requirement and demonstrated phase-information usage.

## MFCFNet

Publication: Zheng et al., IEEE TGRS 61 (2023), article 5212614, pp.1-14; online 21 July 2023; DOI [10.1109/TGRS.2023.3297648](https://doi.org/10.1109/TGRS.2023.3297648), verified in the [university publication record](https://research.lancaster-university.uk/en/publications/multi-feature-collaborative-fusion-network-with-deep-supervision-/). Accessed [repository fulltext](https://eprints.lancs.ac.uk/199400/1/Multi_Feature_Collaborative_Fusion_Network_with_Deep_Supervision_for_SAR_Ship_Classification.pdf): 13-page manuscript with placeholder DOI/acceptance date; pagination below refers to it.

- Sec. III/Fig.1, pp.3-6: DEEP backbone consumes SAR image; HAND backbone consumes its same-size Canny/Harris/Gabor/LBP-derived map, normally one selected feature type. Motivation: preserve expert edge/texture/shape cues while avoiding problematic direct FC-level feature injection and unequal contributions.
- KSCM: auxiliary CE classifiers plus bidirectional probability-matching KL synergy. FFCA: concatenated maps, GAP-MLP-sigmoid channel attention; branchwise sums of attention weights set auxiliary-loss weights. Total Lg+alpha*Ld+beta*Lh+Lk. Spatial dropout=0.5. Heterogeneous ImageNet/SAR-SSL initialization is not specified.
- Evidence: Table II, p.7, five runs: OpenSARShip R18 72.66 to 74.62 with Gabor; FUSAR R18 78.94 to 83.04 with Canny. Table VII, p.11: FUSAR VGG16+Canny equal auxiliary weights 70.54 versus adaptive 84.50, supporting contribution coordination. Table VIII tests synergy removal; ablations select top-gain configurations.
- Sec. IV: separate supervised within-dataset OpenSARShip-1.0 three-class and FUSAR seven-class experiments, training-set three-fold hyperparameter selection; no cross-dataset transfer.
- Inference needs image-derived feature maps, not AIS/complex polarization. Auxiliary classifiers are discarded; HAND encoder remains. Uint8-image compatibility is a methodological inference, not an explicitly tested bit-depth claim.

## Relation to our proposed account

Our intended distinction is the source of useful representation differences: the same available SAR image is encoded through ImageNet-supervised R50 and SAR-SSL R18, and the contribution is justified by measured task complementarity under the declared transfer/support setting. The two papers above establish polarization-guided or handcrafted-view fusion precedents; neither establishes this exact pretraining-source explanation. This is a scoped comparison, not a global novelty claim. The main reader's DSAN++ comparison must decide the architectural overlap.

Do not promote 'two branches', 'complementary features', attention fusion, or uneven branch usefulness as new by themselves. Conversely, MFCFNet means that an image-only/uint8-compatible second branch is not itself a differentiator. DPIG-Net should not be labeled simply dual-encoder, or cited as proving unique phase information. No new experiments were run.
