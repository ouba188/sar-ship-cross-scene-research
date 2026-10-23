# OpenSARShip发布来源、数值单位与近邻方法核实

2026-10-23。只读文献和已有检查记录；本轮不训练、不重读全量像元、不修改实验、不Git。以下实验状态截止主流程交接：1821源对象的数值关系已核实，拟开展尺度补充9头，尚无分类收益结论。

## 结论

**有论文级辐射校准依据，但尚未闭合本地Patch_Cal的精确单位与处理参数。** 原始OpenSARShip论文明确声明校准像元为NRCS sigma0；2.0论文确认校准裁片来自辐射校准产品，却没有重新给出具体单位、标定参数或灰度增强公式。不能仅凭目录名认定本地浮点文件就是未经其他变换的线性sigma0，也不能把它称为平方米RCS。[1,2]

**当前可以称两路输入为“原浮点参照的显示幅度尺度”，不宜称“已核验的绝对物理散射强度”。** 本地关系与论文所述原始幅度/校准/显示分支相容，但数值吻合不是官方源码复现，也不证明跨产品物理一致。

## 官方说明究竟确认了什么

|来源及定位|实际读到的说明|仍缺什么|
|---|---|---|
|Huang等，OpenSARShip，Section IV，作者版第4至5页[1]|原始GRD裁片存VH/VV幅度；MATLAB对原始裁片做灰度增强并存uint8；SNAP 3.0生成校准产品，校准像元声明为NRCS sigma0；伪彩色另由校准裁片经imagesc产生|没有灰度增强代码、阈值规则或完整SNAP参数|
|Li等，OpenSARShip 2.0，Section 3[2]|区分原始幅度、校准、灰度可视化、伪彩色四类；校准裁片从校准产品提取；详细说明指向各场景文件夹文档|没有直接定义Patch_Cal/Patch_Uint8名称到代码的映射；未给8位公式或逐产品校准记录|
|官方DataAndCodes网页[3]|本次实际解析HTML，存在OpenSARShip_1、OpenSARShip_2、original_zip、uint8等数据包链接|未发现可访问的灰度增强/标定源码链接；页面“Codes”字样不等于已经取得代码|
|本地summary.json、product_metadata.csv及主流程FINDINGS|1821对象、22产品、3642极化图匹配；TIFF单位/描述为空；产品XML记录abs_calibration_flag=0|原产品元数据标志不能直接说明派生裁片是否校准；没有取得与这些文件对应的处理图、系数或单位声明|

本轮没有为查说明而下载整套数据包。限定本地OpenSARShip_2根目录查找ReadMe及常见PDF/脚本/文本未找到说明；这不证明发布包从来没有说明。论文[1]明确提及ReadMe.pdf，[2]也指向场景内说明，故应记录为“本轮未取得”，不是“官方没有”。不采用第三方分类仓库的预处理代码来替代官方生成流程。

最重要的生成链区别：**[1]说灰度增强基于原始幅度裁片，不是声称先取校准裁片再开方。** 当两类数值在小窗口内接近比例时，两种生成路线可以产生相同近似关系；因此本次拟合不能识别实际用了哪一路、哪个阈值或何种截断规则。不要把伪彩色imagesc规则移植成灰度PNG的规则。

## 当前拟合关系的准确解释

复用主流程本目录FINDINGS与source_correspondence，不重新估计。源训练1542、验证279；拟合所有像元的clip关系后，四个角色/极化组最大MAE分别为train VH .379699、train VV .377753、val VH .264977、val VV .319715，全部小于.38。取整后逐像元相等比例中位约99.865%至99.929%。这些是源码检查结果，不是论文声称。

令浮点像元为f、灰度像元为p、拟合斜率a>0：

```text
p ~= clip(a * sqrt(f), 0, 255)
t = 255 / a
未饱和处: p / 255 ~= sqrt(f) / t
本轮候选: [log(t_VV), log(t_VH)]
```

这是从已观察关系直接得到的数值解释：t是对应灰度255的浮点开方尺度，不必等于sqrt(max(f))，也不是船体均值、积分RCS或某个已知分位数。原max归一化曾有明显残差，不能再用它描述本轮特征。

- 若同一数值数据换成f'=k*f，拟合斜率变成a'=a/sqrt(k)，则log(t')=log(t)+0.5*log(k)。所以任意未知产品倍率仍可与拟合关系完全相容；低MAE不能排除跨产品倍率差。
- 两个t可以补充PNG没有显式保存的数值尺度，但不能恢复饱和像元的高值差异，量化也使逆变换不精确。
- 若未来确证f为线性sigma0，sqrt(f)才可解释为该后向散射系数的幅度式表示；log(t)本身仍不是平均sigma0，也不是已按dB换算的特征。本轮不要给它附加物理单位。
- 校准定义相同也不等于同一船在不同观测条件下数值不变；论文[1]本身讨论极化、入射角、方向和运动相关差异。校准可比较性与跨采集分类可迁移性不是同一结论。
- 本轮两极化分别估计倍率，不能从各自拉伸后的PNG亮度直接宣称物理VV/VH比值。尺度也可能包含背景、邻船或处理因素，不等同于纯船体因素。

这些边界不阻止已授权的源端小实验：作为有真实输入来源的数值特征可以检验，不必先假装已经取得物理计量证明。推理时需要兼容的浮点文件与灰度配对，仅给PNG不能保证恢复t。

## 三项最接近工作：新增信息和已有思路分开

### Relekar与Shanmugam 2021：处理前尺度特征补充CNN

《Transfer learning based ship classification in Sentinel-1 images incorporating scale variant features》，Advances in Space Research，DOI 10.1016/j.asr.2021.08.042。[4]

本轮实际读出版社摘要、完整可见引言和方法/数据节选；付费完整方法未取得。原文明确以图像处理造成的信息损失为动机，将处理前提取的scale-variant特征补入ResNet50，四类船，OpenSARShip训练/验证及另采数据测试。**这是“CNN以外保留处理前尺度信息”的直接先例。** 但可见方法在辐射校准步骤截断，不能据此编出特征公式，不能断言其特征就是本轮两个显示倍率，也不能断言它没有使用相同强度统计。

与Salerno相比，它更接近当前“已有图像模型另接处理前信息”的组织方式；与Injection相比，它更直接提出信息在预处理阶段丢失的问题。当前可以明确的差异是：冻结现有B，只补两个由本图灰度/浮点配对关系得到的标量，以同信息预算及错配对照检验本图对应作用。没有充分原文依据把“幅度补充CNN”本身称为新思想。

### Salerno 2022：校准幅度进入船轴剖面描述

《Using low-resolution SAR scattering features for ship classification》，DOI 10.1109/LGRS.2022.3183622。[5]

复用主流程已实际读完作者版5页的记录。其校准浮点输入经footprint/惯性主轴和剖面分段，提取幅度、峰和几何特征给随机森林；分段/峰幅度比峰位置数量更有用。它说明幅度信息值得保留，不证明本轮两个尺度已经有效。当前不重建船轴剖面、不依赖可靠首尾或按AIS尺寸误差筛样，也不以其四类随机划分准确率比较本八类跨采集任务。

### Zhang与Zhang 2021：手工描述与深度特征融合

《Injection of Traditional Hand-Crafted Features into Modern CNN-Based Models for SAR Ship Classification: What, Why, Where, and How》，Remote Sensing 13:2091，DOI 10.3390/rs13112091。[6]

复用主流程已读方法及设置至第21页的记录，不声称读完32页。它比较HOG、几何、局部RCS及主轴描述等，讨论L2、MLP映射和末端融合；其OSS1/SLC设置与本地OSS2/GRDH不同，文中未采用OSS2的理由包括复杂背景下船框提取困难。它是特征辅助与融合策略的先例，不能替代本地单位核验。当前只补两个图像级数值尺度、不重新训练CNN、不要求精确船体框，属于受限的信息补充验证，不是首创手工/辐射特征注入。

## 对本轮9头的结论边界

现有证据支持的具体失效位置是：**逐图显示尺度没有显式进入PNG读出**。它比继续改变同一PNG统计更直接增加一个可观测输入，但尚未证明该输入与任务有稳定额外关联。Relekar已给出非常接近的问题动机，Salerno已有幅度描述，Injection已有融合设计；本轮意义应落在真实数值来源、可辨别的替换对照及跨采集适用边界，不落在两个标量拼接的命名。

若正确尺度优于同预算灰度摘要和同采集同尺寸错配尺度，只能先说明本图对应数值在当前协议中有额外作用；不能自动升级为物理散射因果、完整系统胜利或跨产品已校准。若未获得增量，也不能由此否定全部校准浮点信息，因为本轮没有恢复饱和空间细节或改变冻结骨干。按原计划保留结果，不据文献另加网格。

## 阅读与来源凭据

本轮直接读到[1]作者上传全文的数据组织段、[2]作者上传全文的构建/结构段、[3]官方页面及HTML链接、[4]出版社开放部分。没有读取发布增强源码、ReadMe.pdf、逐产品SNAP处理记录或[4]完整付费方法。[5,6]复用已有原文阅读记录，不冒称本轮重新精读。

1. Huang等，OpenSARShip，IEEE JSTARS 11(1):195-208，DOI 10.1109/JSTARS.2017.2755672。作者上传全文：`https://www.researchgate.net/publication/320390622_OpenSARShip_A_Dataset_Dedicated_to_Sentinel-1_Ship_Interpretation`。Section IV是本报告单位及增强分支依据。
2. Li等，OpenSARShip 2.0，BIGSARDATA 2017，DOI 10.1109/BIGSARDATA.2017.8124929。作者上传全文：`https://www.researchgate.net/publication/321407682_OpenSARShip_20_A_large-volume_dataset_for_deeper_interpretation_of_ship_targets_in_Sentinel-1_imagery`。Section 3是格式说明依据。
3. 官方发布：`https://opensar.sjtu.edu.cn/DataAndCodes.html`。本轮实际检查页面，不下载大包，不将第三方仓库当发布代码。
4. Relekar与Shanmugam：`https://www.sciencedirect.com/science/article/pii/S027311772100692X`。仅开放部分可读；题名、作者、DOI及PII另用出版社登记的Crossref元数据核对，元数据不作为完整方法阅读。
5. Salerno作者机构全文：`https://iris.cnr.it/bitstream/20.500.14243/416954/1/prod_468769-doc_189566.pdf`。本地记录为[E:/临时会话/scene_aware_classification_20261018/oss2_distribution_transfer_20261023/next_mechanism_review/SALERNO_READING_ZH.md](E:/临时会话/scene_aware_classification_20261018/oss2_distribution_transfer_20261023/next_mechanism_review/SALERNO_READING_ZH.md)。
6. Injection官方PDF：`https://mdpi-res.com/d_attachment/remotesensing/remotesensing-13-02091/article_deploy/remotesensing-13-02091.pdf`。本地记录为[E:/临时会话/scene_aware_classification_20261018/oss2_distribution_transfer_20261023/next_mechanism_review/INJECTION_READING_ZH.md](E:/临时会话/scene_aware_classification_20261018/oss2_distribution_transfer_20261023/next_mechanism_review/INJECTION_READING_ZH.md)。

本地数值证据为本目录主流程`summary.json`、`product_metadata.csv`、`FINDINGS_ZH.md`及`source_correspondence.csv`。本文没有新增像元审计或分类结果。
