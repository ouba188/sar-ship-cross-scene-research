# M3 task-anchor 有界接口核查

2026-10-30。只读代码与指定近期记录；未运行模型、GPU、训练、测试或 Git。实验、实现与验证由主代理负责。

**结论：在本次核查的 relation_repair 与 reference_delta/coverage/episode 记录中，未见“直接替换原 M3 深层比较锚点，同时保持全部权重及原 value/output 不变”的完整实验。存在近似先例，但不能当作该实验已经失败。**

- `m3_relation_repair_20261029/model.py` 的 `MetricReferenceMatch.descriptors` 保留原始锚点，在读出之后学习深层及浅层通道度量；并非换成 live features。
- `m3_reference_delta_20261030/model.py` 的 `LabelReferenceMatch(live_compare=True)` 已在 `no_grad` 下调用 `super().forward(x, x, ..., dict(reference, anchor=reference['features']), ...)`。但第二次调用只取匹配详情供新增类别证据分支使用；最终仍是第一次旧锚点调用的 `original + correction`，第二次的原 M3 输出被丢弃。`PROTOCOL_ZH.md:49` 明确这一边界，`FINDINGS_ZH.md` 记录未获分类改善。因此：live 比较计算试过，直接驱动旧值回写的本次精确替换未见已做证据。
- `m3_reference_coverage_20261030/METHOD_ZH.md` 的最终中心化仍使用 `anchor_last`；早期误用完整 live layer3 只发生在统计路径，而且该调用没有插入 internal M1/M2，不等于本次 `x`。`m3_reference_episode_20261030/PROTOCOL_ZH.md` 改参考组织及原 M3 训练，不改锚点来源。

上述记录的完整目录前缀为 `E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/`。本结论不扩展为全项目历史穷尽证明。

## 实际接口与影响范围

核读了 `E:/临时会话/sar_m123_current_20261027/sar_m123/` 下三个指定模块及 `selective_backbone.py`。近期实验实际导入路径还需注意：relation_repair 的 `probe.py:15`、reference_delta 的 `run.py:17` 把 **`E:/临时会话/sar_m123_standalone`** 放在 `sys.path` 首位；也已核读该包对应实现，以下定位使用这个实际实验包。

- [spatial_m12.py](/E:/临时会话/sar_m123_standalone/sar_m123/current/spatial_m12.py:38)：`SpatialM12ResNet50 -> BottleneckEvidenceResNet50 -> SelectiveBackbone`。Spatial 子类替换 M1 和 M2/readout pool，但保留 M3 的 `SelectiveReferenceMatch` 和频域读出。
- [bottleneck_evidence.py](/E:/临时会话/sar_m123_standalone/sar_m123/bottleneck_evidence.py:102)：实际 `stage3_paths` 返回 `(prefix, x, ref, raw3, rr3)`。`raw3/rr3` 来自前五个 layer3 block 后的冻结 `anchor_last`；`x/ref` 经当前 `live_last`，包含 pre3x3 的 internal M1/M2 和当前最后 bottleneck 卷积。
- [selective_backbone.py](/E:/临时会话/sar_m123_standalone/sar_m123/selective_backbone.py:78)：继承的 `forward_from` 动态调用上述覆盖方法；传给每极化 M3 的 query 为 `x[:, pol]`，`reference['features']` 为 `ref[:, pol]`，旧 anchor 为 `raw3/rr3`。当前构造只有一个 M3 block，故这里的 `x` 确为 pre-M3。
- [selective_matching.py](/E:/临时会话/sar_m123_standalone/sar_m123/selective_matching.py:34)：仅将 query anchor 改为 `x.detach()`、reference anchor 改为 `reference['features'].detach()`，其余参数原样委托原 `forward`，会仅替换五流中的 deep mean 描述来源。local512 mean、三个 local512 频带描述、mask/mass、pair_valid 和 value 源张量均不变；`descriptors` 本就以 `no_grad` 读出。mass 由 mask/valid 决定，故 value 的 `rm` 分母也不因特征替换而变。

类覆盖语义：只改 `SelectiveBackbone.stage3_paths` 不会命中本模型，因为 Bottleneck 子类已覆盖它。若对既有 M3 实例使用 `operator.__class__ = 新子类`，不会重新执行构造函数，现有参数、buffer、dynamic/frequency/temperature 会保留；须由该子类覆盖 `forward` 并调用原实现。替换模块实例则必须复制旧状态与这些非 state_dict 属性。仅重新绑定模块里的类名不会改变已建实例。这里是接口说明，未实施任何覆盖。

**没有额外组件的参数或结构必须改变，但下游激活必然允许变化。** deep 相似度进入五维 selector，可能改变全部流的 branch_weights，进而改变 Sinkhorn transport、reference 权重、relation logits 和旧 value 的聚合结果。M3 后的 layer4、deep_pool/readout 与 head 因接收新的 `x+update`，输出也可变化；浅层 M2 输入仍是原 `prefix`。

## 三个影响解释的风险

1. **固定权重下的分布失配。** 新余弦进入按旧锚点学出的 selector，且改变旧回写器所接收的聚合值。正负结果回答的是“既有任务特征是否能被既有 M3 直接利用”，不能直接证明/否定任务化匹配可学习性；也不能把所有变化只归于 deep 流贡献，因为五流选择权重会联动。relation_repair 的“关系改善但未兑现分类”正是相关边界。
2. **任务特征来源与冻结边界。** live 同时包含已学 M1/M2 和最后 bottleneck 的作用，收益不能单独归因于某个模块或“修复了语义错位”。`detach` 只截断比较输入梯度，不冻结 value 路径或模型参数；当前 [training_policy](/E:/临时会话/sar_m123_standalone/sar_m123/current/model.py:44) 的 adapt 分支仍会开放三个 LIVE 卷积。若调用它或更新 M3 权重，就已超出本次全权重固定替换，需分开解释；固定现有 eval 端点的探针没有此混杂。
3. **覆盖未命中或重载丢失可伪装成零收益。** 必须作用于实际导入包的已有 M3 实例；上述父类同名方法修改不会生效。当前 [load_model](/E:/临时会话/sar_m123_standalone/sar_m123/current/model.py:24) 按 architecture 重建标准 Spatial 类，state_dict 不保存自定义 Python 类覆盖，因此重载后需重新应用同一行为，否则评价会退回旧锚点。不能把这种结果算作本方案无效。

这是一项可清楚界定的固定参数探针；本说明不预判涨点，也不追加实验、训练或全项目审计。

## 实现到位后的短补充

已只读核对本目录 `task_anchor.py`、`probe.py`、`test_task_anchor.py`。`TaskAnchorMatch.forward` 确实仅以 detached `x/reference['features']` 覆盖深层锚点，再调用原 `forward`；浅拷贝 reference 不改原字典。`attach` 实际采用新建子类实例，保留 channels/dynamic/frequency/temperature、严格加载旧 state，并置 eval/frozen，符合本次接口意图，未发现额外的特征路径修改。此前关于父类覆盖与重载的说明是使用边界，不是已发现本实现有错。

主代理回报：private seed17 的源端及 development 均无 top1 翻转，所有权重不变，3 项 CPU 测试通过。本次仅阅读测试代码，没有重复执行。源同类参考权重质量 VV 由 0.14406 到 0.14879（绝对 +0.00473，即 +0.473 个百分点），VH 由 0.14472 到 0.14939（+0.00467，即 +0.467 个百分点）；类别平衡均值约增加 0.0037–0.0039。该统计在 `probe.source_relations` 中是每个有效槽位的 `reference_weights` 同类和，再按查询/类别平均；不是匹配准确率、最终类别概率，也不是包含 transport/value 后的分类贡献。可表述为“源同类参考权重小幅上移，尚无分类预测收益”，不宜称显著或稳定的类别匹配提升。

此补充时 OSS2 探针按主代理消息仍在运行，尚不能作双数据集结论。若 OSS2 源端也出现改善，主代理拟做的一组配对训练可检验旧回写器能否学会利用该变化；这属于后续验证，不是本次固定权重结果已经证明的收益。本接口核查到此完成，不启动或等待训练。
