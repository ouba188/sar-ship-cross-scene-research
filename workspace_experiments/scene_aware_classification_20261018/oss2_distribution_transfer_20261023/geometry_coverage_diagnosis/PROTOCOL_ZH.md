# 缓存覆盖与量纲诊断

只读取既有cohort、原PNG、coverage、reader及mean/moment缓存，不训练、不改变划分/标签/阈值，不修改其他目录。比较OSS2与当前24港研发分支的实际辅助reader输入；后者全缓存15251行、源/验证/测试按其已有三折取用，不冒称全24港均作为测试。

按已实现weighted_moments定义，mu为每极化每旋转W方向的footprint加权响应均值，sigma为去空间均值与去类公共项后响应的总体标准差，kappa=E[r|r|]/sigma；C4分别统计后平均。这不是标准化偏度或三阶矩。各量以监督W后的logit为单位，直接跨数据集比数值混合了W与激活的差异。

固定输出：全量retained footprint、layer3有效格点/权重Kish格点数；每数据集按cohort顺序等间隔固定128条的实际保留原像素统计（两有效极化、不按标签/预测抽样）；逐折source OOF、同一source行outer、outer validation/test矩分布；仅source OOF定标的descriptor z尺度及预定1%/99%源区间外比例；每类方向W范数及sigma/去类公共W范数、|kappa|/sigma；格点与矩的描述性Spearman及事后真类分层。不会用这些描述阈值重选模型或样本。

同source行outer-vs-OOF用于减少样本组成混杂，仍混合fit数量和具体W/PCA，不等于纯W范数因果实验。Kish格点不代表独立观测数量，深层RF跨格重叠；sigma/W范数仅隔离统一线性增益，不消除方向与协方差变化。两数据集基线不同，不视为单因素迁移实验。
