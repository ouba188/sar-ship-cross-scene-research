# 原生像素双极化两阶段模型

新入口位于 `sar_port.two_stage`：先训练多尺度解耦及空间读出 G，再冻结整个第一阶段，用 `train-base` 单独训练并保存 h0，最后由 `train-h` 复用该 base 检查点训练 H。运行入口为 `python -m sar_port.two_stage.cli`；原 `sar_port.cli` 保留旧 ResNet/M1/M2/M3 对照。

## 已实现的主路径

64 原生裁剪／填充，按完整保护区回退 96、128；双极化独立 ConvNeXt-T 前三阶段。64 输入的原始层为 96×16×16、192×8×8、384×4×4，投影和弱深层注入后均为256通道。互补门控得到 Fc/Fs；六套 G 将 Fc 映射为64通道空间图。第一阶段的类别、场景回归、局部稳定性、类别抑制四类职责有独立梯度边界，D 交替训练。类别抑制默认使用严格受控资格；经验正则须显式选择下述模式。默认先软门控，再逐渐增加稳定性／泄漏权重并进入硬路由。

H 读取冻结的 Fc 和 G(Fc)：Fc 给独立 h0、区域 A 和源端代表选样；G(Fc) 先按几何区域池化，再经各分支 φ:64→128。实际 cut 由几何树按面积优先得到，最多4区，最大原子树8叶。运输边际为真实有效并集面积 μ；A 只产生最终查询权重 `ω=(1−α)μ+α softmax(b+log μ)`。默认 α=.25，初始 b=0。单区直接 ω=1；两个等面积区权重范围 [.375,.625]。

h0 先用源类别 CE 训练并冻结。H 的 φ、β、dustbin、π 和 A 联合优化最终 CE 与匹配 CE；第一阶段和 h0 不更新。原型是源训练集实际船舶观测，每类固定预算、跨港覆盖、真实船身份去重。训练排除同船／同采集组，优先异港支持；真类缺跨港支持时回退到合法源参考。推理不向 H 提供查询类别或港口。

分类基线使用相同骨干、融合和 G，设置 `encoder.gate_enabled=false`，有效支持上精确 Fc=P，走独立纯类别 CE 步骤，不训练解耦辅助任务。与经验解耦比较时匹配补充 query 的身份、次数和顺序。完整 H 必须接续同一经验解耦的已选 E/G 与 h0，冻结三者后比较 H 增量。

## 数据清单

清单可为记录数组或 `{"records": [...]}`。每行至少包含：

```json
{
  "id": "observation_001",
  "label": "cargo",
  "vessel_id": "ship_001",
  "group_id": "acquisition_001",
  "port": "source_port_A",
  "polarizations_coregistered": true,
  "leakage_controlled": false,
  "leakage_regularization_eligible": false,
  "vv": {
    "image": "chips/001_vv.png",
    "alignment_valid": true,
    "axis_reliable": true,
    "ship_axis": {"center": [31.5, 31.5], "length_px": 24, "width_px": 8, "angle": 0},
    "background_mask": "masks/001_vv_q.png",
    "background_validated": true
  },
  "vh": null
}
```

路径相对于清单目录，也支持绝对路径。输入为现有8bit芯片；轴角度为弧度，坐标为原生像素中心。也支持 `detections[target_index].corners` 的 OBB。缺失极化显式留空；`valid_mask` 可标出填充或 nodata。当前实现需要有效 OBB／ship_axis；已有 OBB 但轴不可靠时设 `axis_reliable=false`，保留整船并关闭细分与轴向代价。只有目标掩码而没有 OBB 的接入尚未实现。

配准标记必须来自数据核验。双极化 OBB 不一致时，需要显式确认共同坐标；新入口不会自动把不同芯片坐标当成配准。Q 只能来自验证过的背景掩码，并进一步排除目标保护区；不能把 OBB 外全部像素自动当作海面。`leakage_controlled` 表示受控视图已审计类别与背景、缺失模式、填充和几何等干扰的关联，普通自然观测默认 false。Q 替换构成局部稳定性配对，不能单凭它把该标记改成 true。

`leakage_mode` 默认仍为 `strict_controlled`，由 `leakage_controlled` 及实际合法配对控制类别抑制。`empirical_regularization` 是显式选择的经验类别抑制：独立使用 `leakage_regularization_eligible`，强受控标记可以且在当前 OSS2 中仍为 false。局部 Q 只证明该局部可信背景，不能证明整张 Fs 的背景与类别解关联；D 接近先验也不是因果解耦证明。

经验模式配置 `donor_ids` 与 `supplemental_batches`；二者均只引用源训练行，供体与 query 必须分离。清单中经验资格为 true 的行必须恰好等于补充 query 集合，且覆盖源训练全部类别。每个 query 可用 `background_donor_ids` 指定已核验供体序列，否则用配置中的供体；共同极化、可信 Q 和目标保护仍逐项检查。以下仅为脱敏字段结构，ID 需要替换为实际审核的训练记录，其他必要配置沿用主配置：

```json
{
  "leakage_mode": "empirical_regularization",
  "donor_ids": ["donor_A", "donor_B", "donor_C", "donor_D"],
  "supplemental_batches": [["query_A", "query_B", "query_C", "query_D"]]
}
```

对应 query 行新增 `"leakage_regularization_eligible": true` 和可选 `"background_donor_ids": ["donor_A", "donor_B", "donor_C", "donor_D"]`，同时保留 `"leakage_controlled": false`。只有合格补充批次激活经验 KL/D；自然批次仍进行类别和有效 Q 场景监督。先验从固定补充曝光次数计算，当前四类平衡协议得到均匀先验。基线使用同一补充曝光配置，但只优化 CE。

完整监督检查所选模式的类别覆盖、实际配对和场景参与；`allow_incomplete_supervision=true` 用于明确的不完整监督对照，标记随第一阶段和 H 检查点保留。当前 OSS2 已单独完成数据派生、20 个 Q 审核与经验资格，不修改 legacy 清单或把强受控标记直接置真。实际 1318 条均为64输入；场景纯 Q 格数在双极化三尺度合计为320/0/0，只有 stride4 有直接场景回归监督。原始 sensor nodata mask 仍未知，使用显式存储像素支持约定；该开发实验保留 `training_qualified=false`，另记 `empirical_experiment_eligible=true`。

## 训练与评估

在工程根目录运行。修改 `configs/two_stage_native.json` 中目标港、源验证港；训练港自动取其他港，并通过船舶／采集组隔离。固定强度归一化、场景统计和 h0 RMS 均只由源训练行估计。源验证用于选择检查点；训练过程不读目标港图像。

```powershell
python -m sar_port.two_stage.cli train-stage1 --manifest manifest.json --config configs/two_stage_native.json --output runs/stage1.pt
python -m sar_port.two_stage.cli train-base --manifest manifest.json --config configs/two_stage_native.json --checkpoint runs/stage1.pt --output runs/base.pt
python -m sar_port.two_stage.cli train-h --manifest manifest.json --config configs/two_stage_native.json --checkpoint runs/base.pt --output runs/h.pt
python -m sar_port.two_stage.cli evaluate --manifest manifest.json --config configs/two_stage_native.json --checkpoint runs/base.pt --output runs/base_validation.json --split validation
python -m sar_port.two_stage.cli evaluate --manifest manifest.json --config configs/two_stage_native.json --checkpoint runs/h.pt --output runs/h_validation.json --split validation
```

`encoder.weights_path` 可指定本地 torchvision ConvNeXt-T 权重；默认 null 为随机初始化，不会自动下载。两极化初始化可相同，参数独立。完整第一阶段训练必须覆盖预热、渐增及硬路由阶段；小型链路测试可显式缩短这些步数。

`train-base` 只接受 stage1 检查点，保存独立 base 结果；`train-h` 接受 stage1 或 base。直接传 stage1 仍兼容内部先训 h0 的旧路径；三组增量对照使用上面的独立 base 路径，复用已选 h0，避免重训造成比较对象变化。接续 base 时要求清单、源划分和 H 配置一致，冻结 E/G/h0。

当前 OSS2 三组、三种子训练正在进行，配置和代码已经锁定。固定预算为第一阶段20、h0为10、H为20 epoch；验证 BA 选模。最新进度以[运行状态](<C:/Users/25791/Documents/ChatGPT/cross port sar ship ATR/oss2_three_arm_experiment_20261122/state_local.json>)为准，输入与测试边界见[实验协议](<C:/Users/25791/Documents/ChatGPT/cross port sar ship ATR/oss2_three_arm_experiment_20261122/EXPERIMENT_PROTOCOL_ZH.md>)。历史测试已经曝光，待所有训练完成、配置锁定后统一回归，不用于调参。

检查点保存 E/G、h0、H 参数、归一化和几何配置、源划分、标签映射、真实参考 ID 和表示版本。加载完整检查点不依赖原初始化文件。H 的 CPU 特征目前缓存在内存；参考投影每步用当前 φ 重算，双侧梯度保留。此版尚未提供大数据集的磁盘分块特征缓存或中断续训优化器状态。

`h.batched_matching=true` 默认启用匹配对批量计算，`h.pair_batch_size=64` 控制每块的 query-reference 对数；不是数据 loader 的 query batch size。设 false 可回到逐对路径用于等价对照。批量路径保持面积 OT、方向、禁止边及双侧梯度语义。

确定性原像素预处理另有内容寻址缓存，默认关闭。设置环境变量 `SAR_PORT_PREPARE_CACHE` 为本地目录启用；缓存键覆盖原图/有效掩码/Q 文件内容、记录、输入和几何配置及相关实现内容，输入变化后失效。缓存不包含学习特征，不替代 H 的特征缓存，也不扩大数据或监督资格；只保存张量与基础容器，以 `weights_only=True` 读取。

## 配置与对照

|目标|配置|
|---|---|
|6px主要候选／8px基线|`geometry.min_lengths: [6,16,32]` 或 `[8,16,32]`|
|区域预算|`geometry.leaf_budget: 1..4`，`max_depth: 0..3`|
|关闭深层注入|`encoder.fusion_enabled: false`（参数不创建）|
|注入强度、门控配额|`encoder.fusion_init/fusion_max`、`gate_rho_min/gate_rho_max`|
|纯分类基线|`encoder.gate_enabled: false`，独立纯CE步骤|
|类别抑制模式|`leakage_mode: strict_controlled`（默认）或 `empirical_regularization`；经验模式配置 `donor_ids/supplemental_batches`|
|描述读出|`h.descriptor_mode: frozen_g` 或 `raw_fc`|
|面积／有界区域聚合|`h.alpha: 0` 或 `.25`；α=0不构建A|
|均匀／学习边融合|`h.learn_edge_weights: false/true`|
|同层／同与邻层|`h.adjacent_scales: false/true`|
|原型融合|`h.prototype_weighting: uniform/attention/floor`；保底量 `prototype_floor`|
|几何代价|`h.position_weight/size_weight`，0关闭对应项|
|监督强度及调度|`loss.scene/invariance/leakage`、`warmup_steps/ramp_steps/hard_gate_after`|
|批量匹配|`h.batched_matching: true/false`，`h.pair_batch_size: 64`默认|
|预处理内容缓存|环境变量 `SAR_PORT_PREPARE_CACHE`，未设置时关闭|

几何、OT 与 A 解开后，学习动态粒度暂不启用。各种对照需要各自训练／验证；替换冻结模块的权重不能当作已训练组合的效果。

## 验证

```powershell
python -m unittest discover -s tests -v
```

首版92/92回归记录保留在 `verification_two_stage_20261122.txt`；当前扩展回归为 **110/110通过**，见 `verification/verification_batched_h_all_20261122.txt`。覆盖三种尺寸、空间读出、门控与四监督梯度、冻结、面积/原型隔离，以及新增独立 base、纯CE基线、经验资格/供体调度、批量匹配等价和内容缓存。真实8query批次已核验四项损失非零、门控梯度有限、强受控参与为0；这是接入核验，尚无三组正式涨点结论。历史与当前验证分别记录在 `TWO_STAGE_VERIFICATION_ZH.md`。
