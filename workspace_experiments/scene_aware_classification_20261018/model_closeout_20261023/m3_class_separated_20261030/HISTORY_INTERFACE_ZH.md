# M3 类分离值距离：有界历史与接口结论

2026-10-30。只读核查指定近期历史及实际算子；本文件不代表重新运行模型或测试。未使用 GPU、训练或 Git。

## 核心结论

**已读实现中未发现会使本轮实验无效的接口问题。四组指定历史中未见已完整做过本轮“按参考类别保留实际值特征，并计算 query 到各类对齐原型的距离”的实验。** 这是有界结论，不是全项目穷尽证明，也不是方法原创性或有效性背书。

本轮 `class_match.py` 与 `experiment.py` 符合所述方案：完整旧 M3 保留；新分支读取 pre-M3 query/reference 的 1024 维值，按类别形成原型后计算平方距离；只训练零初始化 C→1024 回写。M1/M2、骨干、旧 M3 和分类头参数冻结。不能将其表述成已替换或修复旧 M3 的全类混合路径：本轮是在旧路径旁新增保留类别的证据通道。

## 历史差异

以下相对路径均位于 `E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/`。

| 历史实现 | 实际比较/回写 | 与本轮的区别 |
|---|---|---|
| `m3_reference_delta_20261030/model.py:63`，`LabelReferenceMatch` | 匹配权重及 transport 质量形成参考权重，再乘参考标签 one-hot，减去合法参考均匀权重对应的类别 prior，回写类别证据 | 使用了合法参考标签，但没有保留各类实际 value 并计算 query–prototype 距离。prior 是合法参考的类别频率，不必然是均匀类别分布 |
| 同文件 `DeltaReferenceMatch` | 加权总 value 减合法参考均匀总 value，仍先跨类别汇总 | 没有类别轴距离 |
| `m3_query_interaction_20261030/interaction.py:46` | query、全类混合 message 或二者逐通道乘积，新增零起点回写 | 已试 query 与实际 value 交互，但 message 已跨类混合；不同于每类独立原型距离 |
| `m3_pairwise_repair_20261030/experiment.py:12` | 在最终 logits 上计算异类 query 配对排序损失，更新原 M3 | 没有类分离 value 路径；类别标签用于训练损失，不能等同为类条件参考距离 |
| `m3_task_anchor_20261030/task_anchor.py:5` | 用 detached 当前任务特征替换深层比较 anchor；旧 value 聚合不变 | 改比较来源，不改全类 value 混合；源同类权重上升、训练预测与 CE 相同不能直接否决本轮值距离 |

已读上述 FINDINGS、相关协议和实际实现。旧 label/label_live 未获得所需分类收益；query-interaction、pairwise 和 task-anchor 也未达到各自完整目标。这些负结果约束“再调匹配权重/普通回写即可改善”的解释，但没有完成对本轮特定值距离的否定。另一方面，仅有代码差异不证明全类混合就是既有误判根因。

## 已确认接口

实际历史导入包为 `E:/临时会话/sar_m123_standalone/sar_m123/`；本次逐行比较其 `selective_matching.py` 与 `sar_m123_current_20261027/sar_m123/selective_matching.py`，文本一致。以下行号指 standalone 包。

1. **归一化。** 原 `selective_matching.py:34-49` 将五流比较描述 L2 归一化；value 则先逐像素通道 LayerNorm、投影到 32 维、再按槽质量求均值。本轮新增 value 不用旧 32 维投影，而是共享相同的 LayerNorm、槽均值和 L2 归一化得到 query/reference 的 1024 维 token。双方在同一表示空间，未混用 protected-anchor 与 task-value 直接求距离。
2. **transport 质量。** 原 `m3.py:21-32` 的 Sinkhorn 包含空匹配项；原 M3 的 moved 除以 offered mass（`selective_matching.py:74-76`），因而保留实配质量比例。本轮 `region_evidence` 除以实际 `row_mass`，形成有效匹配的条件均值，再以原 `reference_weights` 在每类内部重新归一化。它主动去掉此处覆盖率衰减；不是原 moved 的直接重命名。缺支持由 live 标记保留，不能把新原型解释成已保留原回写的全部质量语义。
3. **类内归一化。** membership 仅将参考标签映射到类别轴；`prototype` 聚合的是实际 1024 维 moved 值。按类质量归一化以后再求平方距离，并在有效类别中去均值，最后乘 live。不是 one-hot 概率回写，也没有先求全类均值再假称类分离。
4. **缺失与 mask。** `selective_backbone.py:75-90` 将 query/reference 极化可用性纳入 valid 和 legal；`selective_matching.py:67-73` 再结合槽质量与 pair_valid 得到合法 transport/weights。本轮沿用这些张量，且新槽 mask 同样使用 `minimum(slot, grid_occupancy(valid))`。无支持类 field=0；无合法参考时全部 field=0。内部无效类 distance 的数值本身不作为有效证据，分析必须同时读 class_valid，不能裸取 distance 的 argmin。
5. **身份排除。** `current/data.py:67-71` 排除相同 ID 和双方非空且相同的 MMSI；不在这里排除 pass_group。已有源数据划分的过境隔离属于数据组织，不能归功于 pair_mask。新类分组必须在这一合法掩膜之后进行；现实现满足。
6. **标签及顺序。** 原 `selective_backbone.py:84-86` 构造的 bank 不包含 labels。本轮从实际 refs 取得 labels_ref，attach 时注册 extra_labels，保存/restore 同时恢复参考标签；同一固定 bank 下顺序一致。只使用合法参考标签，query 真值仅进入训练 loss、校准诊断指标或事后评价。若将来换库/重排库，须同步改变此 buffer；当前固定库实验不存在这项错位证据。
7. **梯度和旧结构。** `no_grad` 覆盖新增描述、原型和距离计算，与“本轮只训练 extra_output”一致。configure 仅开放新映射；super.forward 返回的 original 原样保留。gain 固定 0.1，源端 RMS 用于每极化场的尺度校准。推理输出仍经过原 layer4/head。当前新增参数为每极化 C×1024，八类共 16384、四类共 8192；旧 CE 更新 131446 个 M3 参数，故 CE 是同预算训练基线而非等参数量对照。
8. **有限测试证据。** 已只读查看三项 CPU 测试：构造对应关系能区分 aligned/pooled、参考连同标签/transport 同步置换一致、缺失类和全无参考零场。通过结果由主代理报告，本支线没有重复执行。experiment 另记录零起点、关闭新增分支、冻结参数、保存重载和原 chip 前向检查；实际数值以各 fit receipt 为准。

## 对照能回答什么

**本轮最小关键比较是 aligned 对 pooled。** pooled 在每个参考的同一 region 内按槽质量先池化并广播；两臂仍用相同的 query-conditioned reference_weights、合法性和类内归一化。它隔离的是参考 value 的局部空间对齐，不是有无参考、有无匹配、有无参考标签。两臂参数量、训练样本顺序、目标和更新步数一致，场尺度分别从源数据校准。因此它是匹配训练流程下的两种特征设计比较，不是单次固定权重的纯 transport 替换。

若两臂分类逐样本相同而均优于原版，只能说明本轮收益不要求所检验的局部 value 对齐；不能把共有收益归功于 aligned，也不能据此断言局部对齐普遍无效。保留 logit/风险差异，与类别预测收益分开。

相对旧 LabelReferenceMatch，本轮表示差异已由代码确认，但没有同一轮同预算的 label-mass 臂，故不能从本轮均值直接量化“value 距离比标签概率好多少”。不要求补训才可完成当前验证。源查询估计的 class-centered 平均 logit 增量可用于判断共同类别偏移能否复现开发集变化；系数固定为 1，不用目标标签拟合。近似复现只支持该诊断解释，不是唯一物理根因。

## 解释边界与当前状态

原型是单位 reference tokens 的加权平均，代码未将聚合后的原型再单位化。平方距离同时包含方向及原型范数/聚合一致性，不能称为纯余弦相似度；两臂同样如此，本轮不据此追加调参。

本轮与 CrossTransformers 的具体文献关系以主代理已获取 PDF、核读第3-5页式1-3的记录和本目录 PROTOCOL 为依据。本支线未独立完成论文复核，不扩大为严格复现或创新性结论。

截至本次收口，主代理报告两数据集三顺序共12 fit结束。CPU逐查询分析确认两臂OA/BA相同，但OSS2 seed17样本 `pub2_opensarship2_003538` 真值3，pooled预测1、aligned预测2，属于WW差异；不能写成两臂逐样本预测全相同。自建三顺序及OSS2另外两顺序预测相同。原版/CE差异及源共同偏移复现由独立 `analysis.py` 从已保存预测与receipt计算。历史和接口检查到此结束，不反复审阅、不扩大全项目搜索。
