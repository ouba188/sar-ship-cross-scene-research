# References and Verification Record

Verified on 2026-10-27. Stable citation labels for both Introduction and Related Work. The Chinese Related Work is the current deliverable; the earlier English draft is retained and does not yet include R11.

## Citation Map

| Label | BibTeX key | Use |
|---|---|---|
| R1 | huang2018opensarship | OpenSARShip dataset |
| R2 | hou2020fusar | FUSAR-Ship dataset |
| R3 | zhao2022dsan | SAR ship domain-adaptation study; bibliographic verification only |
| R4 | xie2023hogsiam | HOG-SiamShipCLSNet; shared backbone |
| R5 | shao2023dpig | DPIG-Net; VV-guided polarization fusion |
| R6 | he2024cpinet | CPINet; pseudo-Siamese extraction and shared embedding |
| R7 | li2023noncausality | Regional dependence and clutter analysis in SAR ATR |
| R8 | fong2017meaningful | Meaningful Perturbation |
| R9 | hooker2019roar | Remove And Retrain |
| R10 | yin2019fourier | Controlled Fourier-basis perturbations |
| R11 | ross2017right | Explanation constraints used during training |

Introduction replacements: DATASET_REFS = [R1], [R2]; POLAR_REFS = [R4], [R5], [R6]; PERTURB_REFS = [R8], [R9]. Use [R11] only for the connection from explanation to training, not as evidence that our project has implemented it.

## [R1] OpenSARShip

Lanqing Huang, Bin Liu, Boying Li, Weiwei Guo, Wenhao Yu, Zenghui Zhang, and Wenxian Yu. "OpenSARShip: A Dataset Dedicated to Sentinel-1 Ship Interpretation." IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 11(1), 195-208, 2018. DOI: 10.1109/JSTARS.2017.2755672. Primary URL: https://doi.org/10.1109/JSTARS.2017.2755672.

- Verified scope: publisher-deposited Crossref metadata and the dataset authors' official SJTU site, https://opensar.sjtu.edu.cn/ and https://opensar.sjtu.edu.cn/DataAndCodes.html. The official site supplies this citation and distinguishes OpenSARShip_1 and OpenSARShip_2.
- Supports: existence and intended Sentinel-1 ship-interpretation scope of the benchmark. Paired VV/VH use is independently described in the full methods of R5 and R6.
- Gap: the original dataset paper's full text was not obtained in this run. No independent verification of its full construction pipeline, identity filtering, or collection statistics is claimed. Do not substitute OpenSARShip 2.0 statistics for R1; R4 explicitly uses version 2.0.
- Year: use the 2018 journal volume, not the 2017 DOI string as the publication year.

## [R2] FUSAR-Ship

Xiyue Hou, Wei Ao, Qian Song, Jian Lai, Haipeng Wang, and Feng Xu. "FUSAR-Ship: building a high-resolution SAR-AIS matchup dataset of Gaofen-3 for ship detection and recognition." Science China Information Sciences, 63, article 140303, 2020. DOI: 10.1007/s11432-019-2772-5. Primary URL: https://link.springer.com/article/10.1007/s11432-019-2772-5.

- Verified scope: Crossref metadata and the publisher's abstract/preview, returned by https://link.springer.com/content/pdf/10.1007/s11432-019-2772-5.pdf. The returned material explicitly identifies itself as a subscription-content preview; it is not the article's methods full text.
- Supports: a Gaofen-3 SAR-AIS matchup dataset covering varied maritime/coastal scenes, with detection and recognition applications. These basic statements are present in the publisher's own abstract.
- Gap: detailed matching, taxonomy construction, and experimental protocols were not independently read. Requests to the Science China and ScienceEngine routes did not yield the article. No full-method verification is claimed, and no numerical benchmark comparison is used in the draft.

## [R3] Domain Adaptation for SAR Ship Classification

Shuangmei Zhao and Haitao Lang. "Improving Deep Subdomain Adaptation by Dual-Branch Network Embedding Attention Module for SAR Ship Classification." IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 15, 8038-8048, 2022. DOI: 10.1109/JSTARS.2022.3206753. Primary URL: https://ieeexplore.ieee.org/document/9893336.

- Verified scope: complete bibliographic identity from publisher-deposited Crossref metadata; also listed in the references of the independently retrieved R6 publisher PDF.
- Supports in the draft: only the existence of a study explicitly connecting dual-branch attention and deep subdomain adaptation to SAR ship classification. The paragraph does not assert its precise source/target construction or numerical results.
- Gap: IEEE article/PDF routes returned empty or blocked responses, and the ResearchGate route was access-restricted. This run did not obtain a primary full-method text.
- Prior local notes mention optical-to-SAR transfer, R50/R18 branches, CBAM and LMMD. Those details were NOT independently reverified here and are deliberately excluded from the manuscript. Do not cite R3 as proof of a specific cross-port, cross-satellite, or domain-generalization protocol. This is the principal unresolved literature-method gap.

## [R4] HOG-SiamShipCLSNet

Hongtu Xie, Jinfeng He, Zheng Lu, and Jun Hu. "Two-Level Feature-Fusion Ship Recognition Strategy Combining HOG Features with Dual-Polarized Data in SAR Images." Remote Sensing, 15(18), article 4393, 2023. DOI: 10.3390/rs15184393. Primary URL: https://www.mdpi.com/2072-4292/15/18/4393.

- Full-method source retrieved: https://mdpi-res.com/d_attachment/remotesensing/remotesensing-15-04393/article_deploy/remotesensing-15-04393.pdf.
- Read: title/authors; PDF pp. 6-7, Section 2.2.1 and Figures 4-6; pp. 9-11, loss and experimental setup.
- Verified: VV/VH share the feature-extraction backbone parameters. Grouped bilinear operations produce within- and cross-polarization representations at two depths; fused HOG descriptors are concatenated with learned representations. Six predictions are averaged during testing.
- Evaluation scope: OpenSARShip 2.0, four broad classes, 3:1 train/test ratio and 64-by-64 input.
- Supports: existing dual-polarization, hierarchical and handcrafted/learned fusion. This is NOT the differently titled HOG-ShipCLSNet TGRS paper.
- Precision note: the method overview says losses are added, while Equation (13) averages six losses. The draft does not make a claim about this scaling.

## [R5] DPIG-Net

Zikang Shao, Tianwen Zhang, and Xiao Ke. "A Dual-Polarization Information-Guided Network for SAR Ship Classification." Remote Sensing, 15(8), article 2138, 2023. DOI: 10.3390/rs15082138. Primary URL: https://www.mdpi.com/2072-4292/15/8/2138.

- Full-method source retrieved: https://mdpi-res.com/d_attachment/remotesensing/remotesensing-15-02138/article_deploy/remotesensing-15-02138.pdf.
- Read: journal title/authors and PDF pp. 3-8, Sections 2, 2.1, 2.2 and beginning of 3.1. Also read the separate author preprint, https://arxiv.org/pdf/2207.04639.
- Verified: VV magnitude guides VH and product-magnitude branches; cross-attention precedes concatenation; dilated residual dense fusion includes a global VV residual. Experiments use three- and six-category OpenSARShip subsets.
- Sharing: the inspected method text specifies an encoder architecture but does not explicitly establish tied weights across the three encoders. Weight sharing remains unresolved.
- Formula boundary: Equation (1) takes |S_VV times conjugate(S_VH)|. By the magnitude-product identity, this quantity alone does not retain relative complex phase. Do not repeat the authors' complex-data requirement as mathematical proof of phase-sensitive input.
- Bibliographic correction: the 2022 preprint lists Tianwen Zhang and Xiaoling Zhang; the 2023 journal article lists Shao, Zhang and Ke. The draft cites the verified journal article and does not mix author lists or result versions.

## [R6] CPINet

Jinglu He, Ruiting Sun, Yingying Kong, Wenlong Chang, Chenglu Sun, Gaige Chen, Yinghua Li, Zhe Meng, and Fuping Wang. "CPINet: Towards A Novel Cross-Polarimetric Interaction Network for Dual-Polarized SAR Ship Classification." Remote Sensing, 16(18), article 3479, 2024. DOI: 10.3390/rs16183479. Primary URL: https://www.mdpi.com/2072-4292/16/18/3479.

- Full-method source retrieved: https://mdpi-res.com/d_attachment/remotesensing/remotesensing-16-03479/article_deploy/remotesensing-16-03479.pdf?version=1726740972.
- Read: title/authors; PDF pp. 6-12, Sections 3.1-3.4, and the reference list.
- Verified: multiscale fusion; MO-SE uses GAP plus group-bilinear statistics to modulate the opposite polarization; FBC computes local low-rank bilinear codes, applies signed soft thresholding and spatial max pooling; four classification losses are coordinated with GradNorm.
- Sharing distinction: Section 3.1 calls the extractor pseudo-Siamese. Sections 3.1 and 3.4 explicitly place shared parameters in the FC embedding layer; GradNorm uses gradients with respect to this shared layer. This is not evidence of tied backbone weights. Code-level weight tying was not inspected.
- Scope: three- and five-category OpenSARShip classification. Supports prior multiscale/polarization interaction and optimization coordination, not a claim that natural maritime physical factors have been disentangled.

## [R7] Noncausality in SAR ATR

Weijie Li, Wei Yang, Li Liu, Wenpeng Zhang, and Yongxiang Liu. "Discovering and Explaining the Noncausality of Deep Learning in SAR ATR." IEEE Geoscience and Remote Sensing Letters, 20, 1-5, 2023. DOI: 10.1109/LGRS.2023.3266493. Primary URL: https://arxiv.org/abs/2304.00668.

- Full-method source retrieved: https://arxiv.org/pdf/2304.00668v1. Read pp. 1-4, including Sections II-A-C and III-A-D. The unversioned PDF request timed out; v1 succeeded.
- The journal metadata uses "Noncausality"; the author preprint title uses "Non-Causality." Journal authors/year/venue/DOI were checked through Crossref. The final publisher article was not obtained.
- Verified in the author manuscript: baseline Shapley values over target/clutter/shadow; bivariate interactions; true-class pre-softmax score; random clutter rescaling to change signal-to-clutter ratio. Tests concern MSTAR ground targets.
- Supports: region-level dependence, data-associated clutter effects, and targeted input controls. It does not establish that ship neighborhoods are universally irrelevant, or identify a ship image's physical error mechanism. Preprint numerical values are not quoted as final journal results.

## [R8] Meaningful Perturbation

Ruth C. Fong and Andrea Vedaldi. "Interpretable Explanations of Black Boxes by Meaningful Perturbation." Proceedings of the IEEE International Conference on Computer Vision (ICCV), 3449-3457, 2017. DOI: 10.1109/ICCV.2017.371. Primary URL: https://arxiv.org/abs/1704.03296.

- Full-method source retrieved: https://arxiv.org/pdf/1704.03296, served as v4 dated 3 December 2021. Read pp. 1-5, especially Sections 4.1-4.3. Conference identity, authors and pages independently checked through Crossref.
- Verified: optimize deletion/preservation masks for a fixed classifier; define constant, noise and blur perturbations; use total variation, low-resolution masks and random jitter to limit artifacts.
- Supports: explicit perturbations as tests of model dependence and the importance of the intervention definition. Does not establish a new trained model's benefit or identify natural causal mechanisms.
- Gap: the conference-final PDF route attempted here returned 404; methods were read from the author manuscript. Its 2021 revision date does not change the 2017 conference citation.

## [R9] ROAR

Sara Hooker, Dumitru Erhan, Pieter-Jan Kindermans, and Been Kim. "A Benchmark for Interpretability Methods in Deep Neural Networks." Advances in Neural Information Processing Systems, 32, 2019. Primary URL: https://proceedings.neurips.cc/paper/2019/hash/fe4b8556000d0f0cae99daa5c5c5a410-Abstract.html.

- Conference full text retrieved: https://proceedings.neurips.cc/paper_files/paper/2019/file/fe4b8556000d0f0cae99daa5c5c5a410-Paper.pdf. Read pp. 1-4, including Section 3. Official proceedings page confirms metadata.
- Verified: rank inputs by attribution; replace selected pixels in both training and test images with the per-channel mean; retrain from random initialization across removal fractions and compare with random rankings.
- Supports: test-time removal can confound information removal with input-distribution change; fixed-model sensitivity differs from usefulness after retraining.
- Boundary: ROAR is an evaluation protocol, not a claim that every diagnostic intervention requires retraining. Retraining changes the fitted model, and redundant information also affects interpretation.
- No DOI was verified; it is omitted rather than invented.

## [R10] Fourier Perspective

Dong Yin, Raphael Gontijo Lopes, Jonathon Shlens, Ekin D. Cubuk, and Justin Gilmer. "A Fourier Perspective on Model Robustness in Computer Vision." Advances in Neural Information Processing Systems, 32, 2019. Primary URL: https://proceedings.neurips.cc/paper/2019/file/b05b57f6add810d3b7490866d74c0053-Paper.pdf.

- Conference full text retrieved from the primary URL. Read pp. 1-4, including Section 2's Fourier heat-map definition and the opening robustness experiments.
- Verified: real-valued Fourier basis matrices have unit L2 norm; perturbed inputs use a controlled norm and randomized sign; error rates are studied across frequency coordinates. The paper examines robustness trade-offs and use of both high- and low-frequency image information.
- Supports: separate frequency location from perturbation magnitude. These are natural-image experiments, not direct evidence about SAR physical scattering.
- Boundary: a windowed Gaussian scale intervention is not the same operator as a global Fourier basis perturbation. Image-frequency dependence is not physical-factor identification.
- Author spelling follows the conference PDF: Jonathon Shlens. No DOI was verified.

## [R11] Right for the Right Reasons

Andrew Slavin Ross, Michael C. Hughes, and Finale Doshi-Velez. "Right for the Right Reasons: Training Differentiable Models by Constraining their Explanations." Proceedings of the Twenty-Sixth International Joint Conference on Artificial Intelligence (IJCAI-17), 2017. Primary URL: https://www.ijcai.org/proceedings/2017/0371.pdf.

- Official conference full text retrieved. Read printed pp. 2662-2665, especially Sections 1.2, 2.1 and 2.2.
- Verified: binary per-example masks identify input dimensions to discourage; the training loss combines classification, parameter regularization and a squared masked input-gradient penalty. The constrained quantity is the gradient of the sum of log class probabilities. The paper also explores sequential alternative explanations without supplied masks.
- Supports: explanation information can enter a differentiable training objective; this connection predates the present project.
- Boundary: attribution or a beneficial fixed-model perturbation does not automatically supply valid irrelevance annotations. This is not a SAR-specific intervention-supervision method. Our project has not implemented this training constraint.
- Only the first four printed pages were read in this run. DOI and final page were not independently checked and are omitted.

## Verification Limits and Editorial Handoff

This run called web search/open, but the tool returned empty content. Verification therefore used live requests to publisher, conference, author-preprint and dataset-author sites, with PDF extraction in memory. Crossref was used for publisher-deposited bibliographic fields, never as method full text. No old conversation citation IDs are used.

Primary method text was independently read for R4-R6 and R8-R11; R7 was read as an author preprint. The unresolved full-text entries are R1, R2 and R3. R1 has an author-maintained dataset page, R2 has the publisher's abstract/preview, and R3 has bibliographic confirmation only. Detailed DSAN++ protocol claims remain excluded. The final publisher versions of R7 and R8 were not compared with the author manuscripts.

The Chinese draft includes 11 references; the retained English draft includes R1-R10. Neither claims a first method, state-of-the-art performance, physical disentanglement, or verified benefit from our unfinished architecture. The newly reported joint-M1 experiment and prospective intervention-to-training work are not treated as positive method evidence. Same-vessel examples do not isolate platform, angle or port effects when these co-vary.

Only RELATED_WORK_ZH.md, RELATED_WORK_EN.md, REFERENCES.md and references.bib belong to this contribution. No other project files, memory or Git state were edited.

