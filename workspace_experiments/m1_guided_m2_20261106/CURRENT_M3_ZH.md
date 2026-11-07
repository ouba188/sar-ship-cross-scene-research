# 当前工作版 M3：类别证据聚合与回写

2026-11-07。按用户要求同步新版 M3。本文说明已存在的实现和证据，不代表新增训练或八分类工程迁移完成。

## 选用版本

- 当前 OSS2 四分类工作版使用 `Model('aggregate', seed=17)`，加载 `aggregate_17/last.pt`。种子17沿用已有约定，不按评价集排名挑选。
- M1/M2 工作版入口仍是 `current_model.py`，配置为 `current_version.json`。新增的 `m3` 字段仅记录来源，不改变该入口执行行为。
- 实际载入与缓存代码：[run_pilot.py](../m1_role_separation_20261101/run_pilot.py)。
- 模型组装：[model_task.py](../m3_frequency_increment_20261028/source_hard_task_20261101/oss2/model_task.py)。
- 算子：[operator_block.py](../m3_frequency_increment_20261028/class_competitive_matching_20261031/operator_block.py)；空槽处理和辅助分数缩放：[scaled_operator.py](../m3_frequency_increment_20261028/local_scale_repair_20261101/scaled_operator.py)。

## 实际算法

1. VV/VH 分别编码浅层与深层区域描述，生成比较 key 和证据 value。
2. 保留类别条件度量、区域身份、合法参考限制、传输边际及参考权重 beta。
3. 聚合模式将传输矩阵 T 替换为其行列边际外积再归一化，因此取消精细局部 value 对应，但仍计算传输；不是不带匹配的普通均值，也不宣称更省计算。
4. 聚合得到各类别参考表示，与查询 value 比较距离，形成各类相对证据并中心化。
5. 按区域空间权重回写，经每极化输出投影生成残差，加入 layer3 输出；随后进入 layer4 和深层区域读出。
6. 不使用显式频域输入。训练时图像干预属于另一条训练流程，不应混称 M3 频域分支。

## 与当前 M1/M2 的接入状态

当前 M1/M2 实验使用已有 M3 预计算的固定残差：`updated = live_layer3 + cached_m3_delta`。M3参与输出，但不会随新版 M1/M2 特征实时重算参考关系，也没有在最近一轮联合更新。此限制必须随结果一起报告。

旧 `sar_m123_current_20261031` 八分类工程继续保留原 `SelectiveReferenceMatch`。它不是本次聚合版的入口，本次不覆盖其代码或权重。

## 已有证据与边界

| OSS2 队列/比较 | OA差值 | BA差值 |
|---|---:|---:|
| 开发228，聚合M3对同预算无M3，三种子均值 | +2.6316 | +3.4945 |
| 开发228，类别证据回写对普通参考特征回写 | +2.4854 | +3.3923 |
| 扩围严格跨过境125，类别证据对普通参考特征 | +0.53 | +0.59 |
| 扩围378，类别证据对普通参考特征 | -0.97 | -0.62 |

单位为百分点。开发集已用于研发，扩围也不是整个研究的独立盲测。不同对照不能混为“旧完整模型”，不同队列的收益不能相加。

- [三种子配对实验与同权重开关](../m3_frequency_increment_20261028/source_hard_task_20261101/oss2/paired_repeats/FINDINGS_ZH.md)。
- [内部度量和竞争偏置消融](../m3_frequency_increment_20261028/source_hard_task_20261101/oss2/internal_contribution/FINDINGS_ZH.md)：未支持其独立分类优势。
- [证据回写与参考特征回写替换](../m3_frequency_increment_20261028/source_hard_task_20261101/oss2/internal_contribution/writer_review/FINDINGS_ZH.md)：包括真实救回与误伤原图分析。
- [冻结扩围与极化路径](../m3_frequency_increment_20261028/source_hard_task_20261101/oss2/internal_contribution/expanded/FINDINGS_ZH.md)：保留新增378条上的退化。

结论是类别证据回写具有任务内净收益和真实救回，不是精细对应不可替代、所有机制都有益或所有数据集普遍更优。

## 复现范围

仓库发布源码、协议、结果表和诊断图，不发布权重、原始chip、特征缓存和完整训练清单。脚本包含原工作站路径，需要本地数据与对应检查点。当前归档不承诺全新clone后即可运行端到端训练。
