# 原生像素双极化两阶段模型

新入口位于 `sar_port.two_stage`：先训练多尺度解耦及空间读出 G，再冻结整个第一阶段，训练 h0 与 H。运行入口为 `python -m sar_port.two_stage.cli`；原 `sar_port.cli` 保留旧 ResNet/M1/M2/M3 对照。

## 已实现的主路径

64 原生裁剪／填充，按完整保护区回退 96、128；双极化独立 ConvNeXt-T 前三阶段。64 输入的原始层为 96×16×16、192×8×8、384×4×4，投影和弱深层注入后均为256通道。互补门控得到 Fc/Fs；六套 G 将 Fc 映射为64通道空间图。第一阶段的类别、场景回归、局部稳定性、受控类别泄漏四类职责有独立梯度边界，D 交替训练。默认先软门控，再逐渐增加稳定性／泄漏权重并进入硬路由。

H 读取冻结的 Fc 和 G(Fc)：Fc 给独立 h0、区域 A 和源端代表选样；G(Fc) 先按几何区域池化，再经各分支 φ:64→128。实际 cut 由几何树按面积优先得到，最多4区，最大原子树8叶。运输边际为真实有效并集面积 μ；A 只产生最终查询权重 `ω=(1−α)μ+α softmax(b+log μ)`。默认 α=.25，初始 b=0。单区直接 ω=1；两个等面积区权重范围 [.375,.625]。

h0 先用源类别 CE 训练并冻结。H 的 φ、β、dustbin、π 和 A 联合优化最终 CE 与匹配 CE；第一阶段和 h0 不更新。原型是源训练集实际船舶观测，每类固定预算、跨港覆盖、真实船身份去重。训练排除同船／同采集组，优先异港支持；真类缺跨港支持时回退到合法源参考。推理不向 H 提供查询类别或港口。

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

完整监督模式检查合格供体与受控类别覆盖，并记录实际参与数量；无实际配对／场景监督时不会导出“完整方案”检查点。`allow_incomplete_supervision=true` 用于明确的不完整监督对照，标记随第一阶段和 H 检查点保留。当前 legacy 清单仍缺配准／可信 Q／受控视图资格信息，需要补齐后才能启动真实完整训练。

## 训练与评估

在工程根目录运行。修改 `configs/two_stage_native.json` 中目标港、源验证港；训练港自动取其他港，并通过船舶／采集组隔离。固定强度归一化、场景统计和 h0 RMS 均只由源训练行估计。源验证用于选择检查点；训练过程不读目标港图像。

```powershell
python -m sar_port.two_stage.cli train-stage1 --manifest manifest.json --config configs/two_stage_native.json --output runs/stage1.pt
python -m sar_port.two_stage.cli train-h --manifest manifest.json --config configs/two_stage_native.json --checkpoint runs/stage1.pt --output runs/h.pt
python -m sar_port.two_stage.cli evaluate --manifest manifest.json --checkpoint runs/h.pt --output runs/test.json --split test
```

`encoder.weights_path` 可指定本地 torchvision ConvNeXt-T 权重；默认 null 为随机初始化，不会自动下载。两极化初始化可相同，参数独立。完整第一阶段训练必须覆盖预热、渐增及硬路由阶段；小型链路测试可显式缩短这些步数。

检查点保存 E/G、h0、H 参数、归一化和几何配置、源划分、标签映射、真实参考 ID 和表示版本。加载完整检查点不依赖原初始化文件。H 的 CPU 特征目前缓存在内存；参考投影每步用当前 φ 重算，双侧梯度保留。此版尚未提供大数据集的磁盘分块特征缓存或中断续训优化器状态。

## 配置与对照

|目标|配置|
|---|---|
|6px主要候选／8px基线|`geometry.min_lengths: [6,16,32]` 或 `[8,16,32]`|
|区域预算|`geometry.leaf_budget: 1..4`，`max_depth: 0..3`|
|关闭深层注入|`encoder.fusion_enabled: false`（参数不创建）|
|注入强度、门控配额|`encoder.fusion_init/fusion_max`、`gate_rho_min/gate_rho_max`|
|描述读出|`h.descriptor_mode: frozen_g` 或 `raw_fc`|
|面积／有界区域聚合|`h.alpha: 0` 或 `.25`；α=0不构建A|
|均匀／学习边融合|`h.learn_edge_weights: false/true`|
|同层／同与邻层|`h.adjacent_scales: false/true`|
|原型融合|`h.prototype_weighting: uniform/attention/floor`；保底量 `prototype_floor`|
|几何代价|`h.position_weight/size_weight`，0关闭对应项|
|监督强度及调度|`loss.scene/invariance/leakage`、`warmup_steps/ramp_steps/hard_gate_after`|

几何、OT 与 A 解开后，学习动态粒度暂不启用。各种对照需要各自训练／验证；替换冻结模块的权重不能当作已训练组合的效果。

## 验证

```powershell
python -m unittest discover -s tests -v
```

新测试覆盖真实算子的三种尺寸、原像素保留、空间读出顺序、门控配额、四类监督梯度、G冻结、面积守恒、缺极化、1/2原子、A与匹配条件隔离、源原型隔离及完整训练／保存／恢复链路。集成测试使用合成样本，衡量代码与接口正确性，不报告真实 SAR 分类收益。完整检查结果见 `verification_two_stage_20261122.txt`。
