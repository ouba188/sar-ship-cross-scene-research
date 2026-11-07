# 相近研究与 M3 设计边界

2026-11-01。结合代码与只读文献审阅；此文件不宣称已经形成强原创核心。

## 最近工作

- CrossTransformers: Spatially-aware few-shot transfer，NeurIPS 2020：query/support key 建立对应，聚合 value 得到查询对齐类别原型，再求局部距离。与本实现的原型—距离链条直接相关。查询条件原型、key/value 分离和局部匹配不是我们的独有贡献。
- Finding Task-Relevant Features for Few-Shot Learning by Category Traversal，CVPR 2019：类内汇总加类间联合投影，产生任务相关的特征掩码。我们的 own/other 条件是每类分别产生度量，不能声称首次利用类间关系调节特征。
- Few-Shot Learning via Embedding Adaptation with Set-to-Set Functions，CVPR 2020：支持原型集合联合适配。任务条件化本身已有研究。
- Cross Attention Network for Few-shot Classification，NeurIPS 2019：类别原型与查询交叉注意力，残差式增强特征。参考引导的特征更新不是新概念。
- PFENet，TPAMI：支持—查询匹配先验与支持、查询特征参与中间特征增强。任务为分割，但同样限制宽泛的“首次参考证据回写”主张。

## 当前值得检验的具体区别

在极化与区域身份分开保留的前提下，形成类别分别条件化的比较度量；把多个候选类别的相对距离证据，而非直接原始参考特征，回写查询中间层，再由后续骨干加工。

这是具体设计差异，不是自动成立的原创性。需要内部替换实验显示它对当前跨观测任务有实际贡献，并继续与上述方法的匹配适配版比较。

## 代码层面必须准确描述的事项

1. aggregate 仍通过 transport 产生边际和 beta；不能称无匹配的均值基线或默认省计算。
2. competition 给一个查询槽位—参考对的真实列加同一偏置；其主要可作用路径包括真实匹配与 dustbin 的质量分配。各参考独立求解，没有一个跨类别共享的运输预算。
3. beta 随后按类别重新归一化，跨类别 softmax 公共分母会抵消；不能据此宣称类别争夺最终原型权重。
4. 类别齐全时，证据减类均值后接线性层，可由未中心化证据加相应线性权重表达。中心化是去公共分量的结构约束，不是凭空增加表达能力。
5. 当前 shared 输入 [m,m,0] 与 competitive 输入 [mu,other,mu-other] 虽同参数量，有效输入自由度不同。本轮称同结构、同参数预算的条件信息替换，不称严格同表达容量。
6. 普通参考特征回写对照应保留类别、区域和极化组织，不能先合并类别再把损失归因于“缺少相对证据”。还须匹配合理的瓶颈容量与源端消息尺度。

## 原文入口

```text
https://proceedings.neurips.cc/paper/2020/file/68e2d83709f317938b51e53f7552ed04-Paper.pdf
https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Finding_Task-Relevant_Features_for_Few-Shot_Learning_by_Category_Traversal_CVPR_2019_paper.pdf
https://openaccess.thecvf.com/content_CVPR_2020/html/Ye_Few-Shot_Learning_via_Embedding_Adaptation_With_Set-to-Set_Functions_CVPR_2020_paper.html
https://papers.neurips.cc/paper/8655-cross-attention-network-for-few-shot-classification
```
