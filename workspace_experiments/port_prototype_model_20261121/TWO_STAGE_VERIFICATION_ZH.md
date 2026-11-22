# 两阶段实现验证：2026-11-22

## 首版历史验证：92/92

首版 `sar_port.two_stage` 已接通第一阶段、独立h0、H及检查点评估，完整回归 **92/92通过**，耗时77.661秒。命令：`python -m unittest discover -s tests -v`。完整输出在 `verification_two_stage_20261122.txt`；CLI帮助与模块编译检查亦通过。既有ResNet/M1/M2/M3路径未修改。

新增41项测试：编码器9、数据/几何9、H17、原型库4、集成2；其余51项为旧路径回归。

验证包括：

- 原像素裁剪／填充及64/96/128回退；缺失极化、目标截断、可信Q与保护区；双极化有效并集及区域面积守恒。
- 独立双极化前三阶段、软／硬门控、配额、融合关闭、G先空间编码后池化；分类／场景／局部稳定性／交替泄漏监督的梯度归属。
- 1/2/多原子、小船方向等价、禁止边、OT容量；改变A不影响区域、β、运输、方向与π；零描述不改变几何资格。
- 源端真实船去重、缺极化分支距离、跨港覆盖及同船／采集组排除；缺跨港真类支持时合法源支持回退。
- 使用真实ConvNeXt算子的合成三阶段训练，写入并恢复检查点，再评估目标分区；逐参数确认H训练前后E/G完全不变，参考库只含源行，匹配监督实际生效。

检查过程中修复了channels-last硬门控索引、共同Q供体筛选、描述幅值改变合法性、极化并集统计、同港回退及预热调度等实现问题。完整监督额外检查实际受控类别与场景覆盖；无有效支持不能保存为完整H训练。

首版当时读取现有真实清单前4行，确认图像路径与原生尺寸有效，但缺少新入口需要的显式配准／可信Q等资格信息。当时未改写这些标记、未运行真实SAR训练、未产生新准确率结论；需补齐合格数据清单。初版特征缓存在CPU内存。以下追加本轮进展，保留该历史核验范围。

## 当前扩展验证：110/110

当前完整回归 **110/110通过**，耗时32.282秒，命令仍为 `python -m unittest discover -s tests -v`；完整输出为 `verification/verification_batched_h_all_20261122.txt`。中间接口回归记录也保留：`verification/verification_three_arm_interfaces_20261122.txt`为97/97，`verification/verification_empirical_interfaces_20261122.txt`为101/101。不同运行耗时不是同条件模型性能比较。

新增验证覆盖独立 `train-base` 保存和复用、`encoder.gate_enabled=false`的纯CE基线、`empirical_regularization`独立资格与供体/补充query约束，以及批量匹配与内容缓存。默认`strict_controlled`不变；局部Q不自动获得强受控资格。H接续同一个base检查点，完整冻结E/G/h0，保留第一阶段监督模式和实际覆盖记录。

批量匹配默认`h.batched_matching=true`、`h.pair_batch_size=64`，false保留逐对路径。CUDA等价记录见`verification/verification_batched_h_cuda_full_20261122.txt`：该核验配置下最大logit绝对差4.77×10⁻⁷、最大梯度绝对差1.34×10⁻⁷，两路径loss一致。该记录是匹配实现等价与执行耗时检查，不能作为分类收益证据。

`SAR_PORT_PREPARE_CACHE`未设置时仍走原预处理路径且无缓存I/O。缓存核验覆盖真实预处理张量/几何/元数据往返、命中后不重复准备、源文件内容与配置/实现变化失效，以及未激活的缺失文件不新增资格要求。缓存仅加速确定性输入准备，不是学习特征磁盘缓存，不改变训练范围。

## 真实OSS2接入核验与运行边界

实际输入审计保留1318条：训练718、验证227、历史测试373，全部64画布；7个原始目标几何超界样本在三组按同一规则排除。20个Q完成双极化逐图、水域上下文和保护区审核；16核心分为8个独立donor与8个query，余4个用于自然scene。每epoch追加32次query曝光，基线匹配相同曝光。

本轮显式采用`empirical_regularization`，强`leakage_controlled`全部false；只有合格补充query激活经验KL/D。原始sensor nodata mask尚未知，以`stored_native_pixel_footprint`约定标记存储像素覆盖，保留`training_qualified=false`，另记`empirical_experiment_eligible=true`。20个样本双极化纯Q格数为stride4/8/16的320/0/0，只有stride4场景回归有直接监督，不宣称中、深尺度各自接受背景监督。

[真实批次工程记录](<D:/SAR_Experiments/cross_port_sar_ship_ATR/oss2_three_arm_experiment_20261122/real_smoke_summary.json>)显示8个query的cls/scene/inv/leak均非零、门控梯度有限，经验参与8条且四类各2、强受控参与0；该记录的`formal_training=false`。这是正式运行前的接入核验，不证明已经解耦或产生涨点。

随后三组、三种子正式训练已启动，代码与配置锁定，尚无完整三组收益结论。固定seed17/29/43，第一阶段/h0/H分别20/10/20 epoch，以验证BA选模；完整H复用同seed经验解耦的已选base。最新进度见[运行状态](<D:/SAR_Experiments/cross_port_sar_ship_ATR/oss2_three_arm_experiment_20261122/state_local.json>)，解释与评估边界见[固定实验协议](<D:/SAR_Experiments/cross_port_sar_ship_ATR/oss2_three_arm_experiment_20261122/EXPERIMENT_PROTOCOL_ZH.md>)。历史测试已有曝光，不用于调参，不称新盲测；OSS2港口代理也不等同严格跨港留一。本文只记录工程核验与运行状态，不报告尚不存在的正式精度或增量收益。
