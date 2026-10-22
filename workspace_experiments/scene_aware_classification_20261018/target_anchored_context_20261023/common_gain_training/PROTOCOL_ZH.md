# 共同增益限制：训练匹配对照

2026-10-23，训练前固定。只写本目录，不改父级、不Git。

## 范围与预算

沿用old父own_regions_full和new partition_replication各自已冻结q/oq/env/base/oof及六折划分。5695观测、4312 MMSI、163过境、15港、三类；不是八类新队列。old selected_kind为原源选择读出，new固定layer3。每折的inner OOF selected_kind沿用现有固定结构，不重新拟合或选择任何读出。

每队列6fold×3seed×2arm，共72头；seed17/29/43，200step，AdamW(.003,.01)，原类别/MMSI权重CE，每5步源val balanced NLL选步、含0步、改善阈值1e-6。零初始化最后线性层，网络79->32->32，与父相同3616参数；所有目标全量激活。仅这一轮，不扩大参数范围或重复换seed，不加weighted-neutral训练或whole/fg/bg臂。

## 两臂

- contrast_only_gate：a=.5*tanh(net(env))；只在结构有效组减去算术均值，d=a-mean(a)，增量Σd_g*q_g。有效组系数和为0；倍率理论范围(0,2)，数值饱和允许端点近似。
- range_matched_unrestricted：d=tanh(net(env))，不减均值，增量Σd_g*q_g，倍率同(0,2)。该对照用于不把扩大门控范围误当作零共同分量结构的优势；同范围不意味着同一参数化或相同可达系数集合。

layer3-only有效组16..31，stem+layer3有效组0..31，不看逐样本q阈值。deep-only前16槽严格零，不参加均值。测试forward用outer q，训练用源OOF oq/oof；env仍是外层源投影，所以不是整流水线OOF。A分数直接来自缓存，始终不变。

## 验证与分析

训练前合成检查专门检测将deep-only零槽错误计入均值；训练后检查全部样本有效组sum(d)的float32误差、无效槽零、零q零增量、倍率范围、0步回退及保存头重放。保存72PT、全部预测、救回/误伤、逐折/汇总/class、数据来源、曲线和阶段时间。分析CPU独立回放对CSV。

参照包括各队列A、已完成父原模型和common_gain_diagnosis固定头contrast-only；不重训参照。事后条件区间按pass/MMSI分别计算、不择优，训练seed不当独立数据。

原图按每队列seed29 contrast_only_gate相对A的救回/误伤/CC/WW首ID选例，优先无外部邻域；若某类不存在则报告而不捏造。原尺度VV/VH复用native PNG，展示原A、父原模型、contrast及range对照的真实分差，不插值或提新数据。

## 贡献边界

主流程已在RELATED_WORK_AND_CLAIM_ZH.md记录GCT与GRN等原文方法。通道竞争、相对归一化、1+tanh、恒等初始化、减均值不作为新发明；本支线未重新阅读这些论文。当前候选仅是针对已观测冻结SAR读出共同倍率/类别参照纠缠的限制和证据链，不预定有效或创新结论。

controller_readout_probe已完成72头，whole/fg与双区配对CI跨0，不支持双区独立优势。本轮固定原env是为了隔离门控结构，不能借其结果重新宣称区域分解独立有效。八类baseline/OOF由另一支线推进，本轮不触碰。两臂若都无收益则保留负结果与固定模型诊断，不反复调参。
