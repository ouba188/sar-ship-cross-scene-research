# 任务式强单 R50 多数据诊断：实现、验证与解释边界

日期：2026-10-10。独立入口：`E:/Hermes/paper_tracking/sensitivity/oof_reliability_check/strong_single_multidata_explain.py`。

后续主线补充：最终CLI的固定单路集合已预先加入`r18_control`中的`r18_task_trained`，对接前核对相同样本与支持抽样；不是加入双路拼接/融合臂。下文OSS2验证数字对应早先不含R18的集合，保留原结果，不能与最终扩展集合的共同难例计数混用。R18对照全部25任务已完成，详见`r18_control/FINDINGS_ZH.md`。

脚本已实现；已用 OSS2 的 20261010/11/12 三个神经 seed、原五组 64 标签支持集完成真实验证。23 LOPO + F 的新训练正在主任务中进行，本说明不将它们标为已完成。没有修改训练脚本、主报告或记忆；没有联网、GPU 运算、像素提取、新训练或新四格交叉。生成数据仅写入本说明所在目录的独立解释子目录。

## 1. 输入契约

新训练布局：

```text
E:/Hermes/paper_tracking/sensitivity/strong_single_multidata_20261010/
  targets/{dataset}/{task}/seed_{seed}/predictions.npz
```

`dataset` 为 `self24ports`、`f0921`；也支持已有 `oss2`。外部任务名为 `f0921`、`oss2`，LOPO 任务名为对应港名。默认要求三个指定 seed 都有 `COMPLETE.json` 和 `predictions.npz`，才分析该任务。已发现但未齐备的任务写入 `INPUT_INVENTORY.json`，不将少数 seed 当完整结果。尚未创建目录的任务不会被发现，因此总覆盖量还必须看 `RUN_RECEIPT.json` 中的实际任务数，不能把成功退出解释为 24 个新任务全部完成。

| 键 | 形状/含义 |
| --- | --- |
| `arms` | 包含 `frozen_single_same_readout`、`task_adapted_single_same_readout`；按名称索引，不依赖顺序 |
| `predictions` | `[draw, arm, N]`；当前 `[5, 2, N]` |
| `margins` | 新格式 `[5, 2, N]`，同原分类分数的 top1-top2 差；优先读取 |
| `rows`, `ids` | N 个查询队列样本的 manifest 行号、样本标识 |
| `support`, `support_labels` | `[5, 64]`，support 是队列内局部索引 |
| `eval_masks` | 布尔 `[5, N]`，逐 draw 排除支持样本 |
| `source_rows` | 允许的源端行号；同任务各 seed 必须相同 |

新格式有 `margins` 时无需加载 PyTorch、特征或检查点。旧 OSS2 文件不含分数/差值，脚本用已有 `*_phi.npy`、`final_tail.pt` 和原 `aligned_residual_ops.residual_scores` 在 CPU 回放，lambda 固定 8，原 float32 运算次序不变；两臂全部分类必须与存档严格一致。显式保存的 `logits`/`probabilities` 也支持 `[draw, arm, N, 8]`，但不猜测一个无单位的 `scores` 键。

**单位修订：新 `margins` 是原 logit 差，不是历史 dual 报告的 softmax 概率差。** 本脚本固定使用 `<0.05`、`[0.05,0.20)`、`>=0.20` 分层，明确记录 `original_logit_gap`；不能与历史概率差档位直接横比，也不能视作校准正确率。若输入明确是概率，则标记 `probability_gap` 并分开汇总，不混合单位。所有阈值只作描述，不据目标查询结果选门控或新配置。

manifest 默认复用已有 self24ports 源 manifest 及两个 transfer 的 `target_manifest.csv`。严格核对行号对应的 ID、支持标签，以及相同队列/支持/评估掩码；不重排、截取交集或默默丢弃不匹配样本。LOPO 按 manifest 的 `port`、外部集按 `port_id` 另行输出分港结果；未知港仅标 `unknown`，不推断地理身份。

## 2. 固定比较方向与差距桥接

`dual_vs_trained`：候选是固定当前双路，参考是该 seed 的训练 compact 单路。rescue = 双路对、单路错；harm = 双路错、单路对。`trained_vs_frozen`：候选是训练单路，参考是同 seed 冻结 compact 单路。两种方向明确分列，不混用“救回谁”。双路引用 `backbone_ablation_20261010/targets/{dataset}/{task}/predictions.npz` 的 `dual_refit_late`，不是每个神经 seed 新训一个双路。

逐任务、seed、draw 同时计算 OA/BA：

```text
D - T = (D - R) + (R - F) + (F - T)
D = 固定当前 dual_refit_late
R = 同一匹配存档的 current full R50-only: r50_late
F = frozen compact
T = task-trained compact
```

每一行验证恒等式，单位是百分点。第一项是已有匹配设置下的第二路增量，第二项是完整 R50-only 与冻结 compact 的配置差异，第三项是任务训练影响的反向量。第二项包含完整读出、几何及其他设置差异，不能命名为某个孤立模块的因果效应；三项都是配置差异，不是独立因果分解，更不能把全部 D-T 归于双骨干数量。

聚合顺序：各 seed 内先均匀平均支持 draw，再等权平均任务，最后算神经 seed 均值与样本 SD。23 个 LOPO 港不会按样本量加权。BA 按当前评估子集实际出现类别计算；逐类 BA 贡献按该 draw 类别数归一化，缺类任务不会被当成有该类。逐类贡献之和必须还原宏平均 BA 差距。

## 3. 稳定性与已测配置共同难例

- `seed_consistency_queries.csv.gz`：每行一个 query/draw，给出各事件的神经 seed 票数、全 seed 一致救回/误改。计数不再乘以 seed 数。
- `seed_consistency_unique_queries.csv.gz`：按 query 去重，记录实际作为查询的 draw 数、任一/所有有效 draw 上的事件及跨 draw 反转。只对实际进入评估的 draw 作全称判断，支持样本不参与。
- 同一 query/draw 的双路正确性固定，因此相对固定双路，跨神经 seed 不可能同时出现 rescue 和 harm；可能出现的是“全 seed 一致”与“仅部分 seed 发生”。跨支持 draw 双路可以改变预测，因此可以出现方向反转。
- 预先固定的单路配置集合包含全部指定 seed 的训练 compact、冻结 compact、当前 full R50-only，以及 `dual_reconstruction_20261010/.../predictions.npz` 的 `mlp_late`。缺失或错配任何必需存档就报错，不悄悄缩小集合。固定集合未按目标表现择优，MLP 部署时不运行 R18，但训练使用 R18 教师。
- `commonhard_queries.csv.gz` 记录这些单路配置全部错误时双路能否救回，也记录“双路错、至少一个单路对”以及“双路错、全部单路对”。重复冻结预测只是逻辑集合成员，不增加独立证据。另有去重 query 表。

准确表述是“已测配置共同难例”。不能据此宣称任何单骨干都不可能解决，不能把样本重叠的五个支持 draw 或三次复用的固定双路作为独立重复实验。

## 4. 可用功能路径证据

新任务的四格交叉由主任务负责，本脚本不求解新交叉臂。若任务目录已有 `diagnosis/interactions_by_seed_draw.csv` 及对应 `seed_*_predictions.npz`，验证队列和两个原始对角线后引用；缺失时记录 `unavailable`。

已有 dual 诊断的 `scores.npy`、`expert_delta_parts.npy` 可直接 mmap 读取，无须重新提取或重建核。先核对 DD_late 与固定双路预测一致，再将下式的真类减固定 RR 最强错误类的 margin 变化，按本轮 rescue/harm/both-correct/both-wrong 分组：

```text
historical DD expert - historical RR expert
  = prior_delta + old_relation_reweight + kernel_direct
```

这仍是历史 DD-RR 专家 logit 的代数记账，只是在本轮事件集合内描述其变化；**不等于双路最终概率减训练 compact 分数，不构成 D-T 的分数分解**。逐类别分数之和经存档验证，softmax 非线性不被忽略。

旧 OSS2 回放同时可得训练/冻结先验，另外导出 `trained-minus-frozen = source_prior_delta + support_correction_delta`。后者同时包含关系和残差系数的变化，不冒充纯核贡献。新格式只含 margins 时不为此重读大特征，缺失这组分解不影响救回、稳定性、桥接和共同难例输出。

## 5. 已完成验证与真实数值

输出位置：`E:/Hermes/paper_tracking/sensitivity/strong_single_multidata_20261010/explain_oss2_validation/`。

真实 OSS2 三 seed、五 draw、每 draw 3,681 个查询。两臂所有 112,350 个存档分类在原 PyTorch 环境 CPU 回放完全一致；支持/行号/ID/掩码对齐、桥接恒等式、逐类贡献加和均通过。旧四格交叉的两个对角线也与本轮一致。

| 指标 | D-R | R-F | F-T | D-T |
| --- | ---: | ---: | ---: | ---: |
| OA pp | +10.06792 | -2.08639 | +1.19171 | +9.17323 |
| BA pp | +4.42276 | +0.90884 | -1.07944 | +4.25216 |

BA 差距的神经 seed SD 为 0.86697pp；固定双路本身不是三次训练。完整 R50-only 的 OA 低于冻结 compact 而 BA 较高，已经说明“完整读出必然普遍更强”并不成立。任务训练相对冻结 compact 的 OA -1.19171pp、BA +1.07944pp，也不能简写为训练完全失败。

| 描述 | 数量 |
| --- | ---: |
| query/draw 曝光，跨 seed 不重复 | 18,405 |
| 双路对训练单路，全 3 seed 一致救回 | 2,009 |
| 双路对训练单路，全 3 seed 一致误改 | 622 |
| 所有有效 draw 与所有 seed 均被双路救回的去重 query | 12 |
| 所有有效 draw 与所有 seed 均被双路误改的去重 query | 4 |
| 固定已测单路集合全部错误的 query/draw | 7,184 |
| 上述共同难例中双路救回 | 888 |
| 上述共同难例中双路也错 | 6,296 |
| 双路错而至少一个已测单路对的 query/draw | 4,269 |

后四行是同一固定比较集合的描述性统计，不是模型选择后的优势估计。888 个共同难例救回不能掩盖 4,269 个反向事件，也不能与它简单相减当双路 OA 优势，因为“至少一个单路对”是一个使用真值的逻辑并集，不是可部署分类器。

原 logit gap `<.05`、`[.05,.20)`、`>=.20` 三档，双路相对训练单路的净 OA 分别为 +12.13485、+10.65417、+6.77925pp；对应 rescue/harm 为 2267/1108、4552/2324、3406/1728。此处合计 55,215 次 query/draw/seed 曝光，含重用查询和固定双路，不是 55,215 个独立样本。分层呈描述性边界效应，不证明目标门控有效。

额外验证：`--self-test` 使用小型临时数据，覆盖错配 ID 拒绝、正反 rescue/harm、缺类 BA、空档、阈值边界、seed 投票、差距桥接、两个不同类别覆盖的任务、仅 margins 无特征的输入、训练中任务排除和固定集合 common-hard。没有新增持久测试数据或测试框架。

运行环境差异保留为失败记录：最初用 ClearSAR 的 PyTorch 2.7.1+cu128 在 CPU 回放，seed 20261011 的 draw2/局部 row2474 出现 1 条不一致（实际0、存档3），脚本按要求停止；切回原诊断环境 PyTorch 2.12.1+cu132 后三 seed 全部精确。证据确认环境切换能消除该差异，尚未隔离具体底层算子原因，不直接断言仅为舍入。本轮没有改原预测或放宽一致性标准。环境名称含 CUDA 不代表使用 GPU，本脚本检查点始终 map 到 CPU。

## 6. 最少下一步与未证实边界

先完成已经固定的 23 LOPO + F 对比及主任务已有的先验/核交叉，分数据集判断桥接项，不把 OSS2 直接外推。

1. 若任务训练造成的关系迁移反转仍成立，最有判别力的一项新训练干预是：保持同一单骨干、源任务训练、seed、预算和源端选择，加入固定源端支持关系保留约束，与原目标比较。只看它能否减少核路径负效应同时保留先验收益；部署仍用一个骨干。约束形式/权重只能由源端选择，不按目标数据集挑最优。现有同检查点源/目标交叉已排除“仅因全源 refit”的充分解释，无须重复同一控制。
2. 若 R-F 而非 D-R 占主要差距，优先完成同一 R50 下的完整读出匹配对照；先核实到底是哪项读出配置在改善或损害表现，再归因结构。是否值得额外分支还需相应实际成本，当前脚本没有测成本。

已证实的是当前存档配置的配对效果、净纠错、跨 seed/支持抽样稳定性，以及可核对的功能路径和配置桥接。未证实的是某种物理散射机制、独占信息、双路必要性、任意强单不可能、预训练/输入/架构因素独立效应、全网络最优性。F 历史无标签 SSL 暴露和研发数据复用仍适用。不能用这个报告替代任务训练后的 SSL-R18 单独分类对照。

## 7. 调用与产物

PowerShell，已有 OSS2 验证：

```powershell
& 'E:/Hermes/paper_tracking/envs/msfa-openmmlab-py310-rtx50-cu132/Scripts/python.exe' 'E:/Hermes/paper_tracking/sensitivity/oof_reliability_check/strong_single_multidata_explain.py' --root 'E:/Hermes/paper_tracking/sensitivity/strong_single_r50_task_20261010' --out 'E:/Hermes/paper_tracking/sensitivity/strong_single_multidata_20261010/explain_oss2_validation'
```

新 24 任务完成后，原 schema + 新 margins 可直接调用。只分析所列三个 seed 已齐备的任务，不等待或启动训练：

```powershell
& 'E:/Hermes/paper_tracking/envs/msfa-openmmlab-py310-rtx50-cu132/Scripts/python.exe' 'E:/Hermes/paper_tracking/sensitivity/oof_reliability_check/strong_single_multidata_explain.py' --root 'E:/Hermes/paper_tracking/sensitivity/strong_single_multidata_20261010/targets' --out 'E:/Hermes/paper_tracking/sensitivity/strong_single_multidata_20261010/explain'
```

`--root` 可同时给两个根以合并新任务与旧 OSS2；重复 dataset/task 会报错。`--seeds` 可指定种子集合，报告准确记录数量，不将单 seed 描述为跨 seed 证据。`--self-test` 单独运行测试后退出。非标准布局可用 `--task-specs` JSON 数组，字段为 `dataset/task/directory`，可选 `manifest/id_column/port_column/reference/reference_mode`；其他键不需修改训练产物。例：

```json
[{"dataset":"oss2","task":"oss2","directory":"E:/Hermes/paper_tracking/sensitivity/strong_single_r50_task_20261010"}]
```

重点产物：

- `NUMERIC_FINDINGS.md`：自动数字报告，含桥接、稳定性、共同难例、分层和边界。
- `gap_bridge_by_seed_draw.csv`、`gap_bridge_summary.csv`：每项实际指标、三段差值和闭合误差。
- `rescue_harm_by_seed_draw.csv`、`per_port.csv`、`per_class.csv`、`class_macro_ba_contributions.csv`：净效应、反向误改、支持类覆盖和 BA 贡献。
- `task_seed_means.csv`、`task_summary.csv`、`dataset_seed_means.csv`、`dataset_summary.csv`：任务宏平均和神经 seed 波动。
- `confidence_strata.csv`、`confidence_exposure_summary.csv`：固定原分数差分层及各档分母。
- `seed_consistency_by_draw_class.csv`、`unique_query_summary.csv`、`commonhard_by_draw_class.csv`：稳定性/共同难例计数；样本 ID 和去重明细在 `tasks/{dataset}/{task}/*.csv.gz`。
- `functional_parts.csv`、可用时的 `existing_prior_kernel_interactions.csv`：带明确比较对象和单位的功能证据。
- `INPUT_INVENTORY.json`、`RUN_RECEIPT.json`、逐任务 `RECEIPT.json`：实际覆盖、路径、单位、运行环境和证据缺口。

## 8. 独立补充：任务训练后的 SSL-R18 单独分类是否已有

**本轮只读核查未发现与当前 23 LOPO/F/OSS2 队列、同 5x64 支持协议匹配的“任务训练后 SSL-R18 单独分类”结果。不能用现有 expert 单头替代。** 核查了当前 sensitivity 代码、协议/汇总/说明，以及 `visual_reliable_baseline` 中的 R18/SSL-only 入口；这是当前项目已定位产物的结论，不声称穷尽所有磁盘上的未知别名文件。

有两个容易混淆的现成项：

1. `E:/临时会话/visual_reliable_baseline/b1_clean_expert.py` 确有 `ssl_only` 分支，`b1_clean_expert.json` 也确实存在，键为 `im+ssl/ssl_only/verdict`。它读取 `e206_out/ssl_resnet18.pt`，未经当前源任务残差训练；使用旧 4,962 裁片港口/时间协议和 rank-minus-random 汇总，不是当前任务训练后 R18、五组 64 支持、全目标八类 OA/BA 对照。因此不能报告为本轮 R18-only 已完成。
2. 当前 `backbone_ablation_core.py` 只提供 base、dual_original、dual_refit、r50 的 expert/late/unified 模式，没有 R18-only。`compact_fusion_source._expert_inputs` 显式拼接 R50 PCA128 与 SSL PCA64；expert 核也同时使用 R50/SSL 的补空间。因此 `dual_refit_expert` 只是单个分类头，不是一个 R18 骨干。

### 8.1 每个 LOPO 任务的已训练 R18 特征

以下 `{port}` 对应 23 个实际外留任务。512 维源/目标特征均已存在，不需要再提特征或训练网络。

```text
source_extra 等价缓存：
E:/Hermes/paper_tracking/sensitivity/evidence_aware_handoff_20261008/source_features/{port}/features.npy

上述源特征逐行的全局 source rows：
E:/Hermes/paper_tracking/sensitivity/evidence_aware_handoff_20261008/source_features/{port}/rows.npy

同任务另一个相同行号存档：
E:/Hermes/paper_tracking/sensitivity/protected_drift_activation_20261008/ports/{port}/source_rows.npy

target_extra 等价缓存：
E:/Hermes/paper_tracking/sensitivity/module_completion_20261008/ports_projection/{port}/matched_joint/joint_extra.npy

目标全局行号：
E:/Hermes/paper_tracking/sensitivity/protected_drift_activation_20261008/ports/{port}/target_rows.npy

原目标 ID / 支持 / 标签 / mask / source_rows 的统一入口：
E:/Hermes/paper_tracking/sensitivity/learned_frequency_residual_20261009/tasks/self24ports/{port}/target.npz

任务训练后的 layer4 权重：
E:/Hermes/paper_tracking/sensitivity/module_completion_20261008/ports_projection/{port}/joint/layer4.pt
```

源 `features.npy` **已经按 `rows.npy` 子集排列**，不能再用全局 source rows 直接索引它。目标 `joint_extra.npy` 已按该任务 target rows 排列，也不是全源 manifest 的矩阵。`source_features/{port}/COMPLETE.json` 明确记录所用任务训练后 `joint/layer4.pt`；其 `training:false` 只描述随后那次缓存提取没有训练，不表示权重没有经过任务训练。

已逐个只读确认全部 23 任务：源两份 rows 与 `target.npz.source_rows` 一致，target rows 与 `target.npz.rows` 一致；源/目标特征首维分别等于对应行号数，末维都是 512，记录的 layer4 文件存在。没有读全部特征值或执行神经前向。

| 港任务 | 源行数 | 目标行数 |
| --- | ---: | ---: |
| Antwerp-Bruges | 36544 | 886 |
| Busan | 36849 | 871 |
| Callao | 37485 | 476 |
| Fujairah | 35746 | 1158 |
| Hamburg | 35889 | 1619 |
| Houston | 37093 | 804 |
| Jebel Ali | 31937 | 4602 |
| Los Angeles | 37372 | 583 |
| Mombasa | 37799 | 149 |
| New York | 37548 | 470 |
| Newcastle | 37495 | 299 |
| Ningbo-Zhoushan | 30601 | 5756 |
| Port Hedland | 36693 | 859 |
| Port Klang | 32669 | 3425 |
| Port Said | 35603 | 1895 |
| Qingdao | 34902 | 1981 |
| Richards Bay | 36742 | 744 |
| Rotterdam | 36929 | 573 |
| Santos | 35808 | 1658 |
| Shanghai | 33881 | 2390 |
| Singapore | 30246 | 5259 |
| Sydney Botany | 37484 | 304 |
| Tanger Med | 36319 | 1266 |

### 8.2 F_0921 与 OSS2 的实际路径

F_0921：

```text
source_extra: E:/Hermes/paper_tracking/sensitivity/joint_residual_pilot_20261008/joint/source_extra.npy
target_extra: E:/Hermes/paper_tracking/sensitivity/joint_residual_pilot_20261008/joint/target_extra.npy
source rows: E:/Hermes/paper_tracking/sensitivity/f0921_transfer_20261007/source_split.npz [source_rows]
target metadata: E:/Hermes/paper_tracking/sensitivity/learned_frequency_residual_20261009/tasks/f0921/target.npz
layer4: E:/Hermes/paper_tracking/sensitivity/joint_residual_pilot_20261008/joint/layer4.pt
```

F 源矩阵 `(38091,512)` 是**全源 manifest 顺序**，必须用允许的 29,717 个 `source_rows` 切片；不是所有 38,091 条都可用。目标矩阵 `(123489,512)` 与目标 `rows=arange(123489)` 对齐。已核对 source split 与目标设计中的 source rows 完全一致，且与 excluded_source_rows 不相交。

OSS2：

```text
source_extra 实际缓存: E:/Hermes/paper_tracking/sensitivity/compact_fusion_20261010/cache/oss2_source_extra.npz [features, rows]
target_extra: E:/Hermes/paper_tracking/sensitivity/module_completion_20261008/oss2/joint/target_extra.npy
source rows: E:/Hermes/paper_tracking/sensitivity/opensarship2_transfer_20261007/source_split.npz [source_rows]
target metadata: E:/Hermes/paper_tracking/sensitivity/learned_frequency_residual_20261009/tasks/oss2/target.npz
layer4: E:/Hermes/paper_tracking/sensitivity/module_completion_20261008/oss2/joint/layer4.pt
```

OSS2 的 `joint/source_extra.npy` 不存在，但上述缓存确实存在，`features` 为 `(36996,512)`，**已经是合法源行子集**；其 `rows` 与 source split 完全一致，不能再按全局行号二次索引。目标 `(3745,512)` 与 `rows=arange(3745)` 对齐。`compact_fusion_source._external_features` 代码表明缓存来自该任务 `joint/layer4.pt` 的冻结提取；本次只读检查没有调用其可能执行前向的缓存生成分支。

源标签统一读取 `E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916/manifest.csv` 的 `class_id`，再按允许 source rows 切片。F/OSS2 目标标签分别在各 transfer 目录的 `target_manifest.csv`，目标适配必须继续使用上述 `target.npz` 中原五组 support/support_labels/eval_masks。

### 8.3 这个缺口能回答什么

后续最直接的独立控制应使用上述**同一个已任务训练后的 R18 表示**，仅凭它的 512 维特征构建源类别先验和目标支持适配，去掉所有 R50 特征、R50 基础预测和 R50 补空间，再比较固定双路。源头/归一化必须匹配允许源行，不能直接拿双路的拼接先验当 R18-only 先验。

这能判断当前双路收益是否只是“R18 自己已足够”，或是否需要 R50/R18 结合。该 R18 本来是在双路任务上下文中训练的，因此这仍是“现有任务训练后 R18 的单独读出”控制，不等于独立优化到最佳的 R18 单骨干，更不证明任意单路能力上限。

按用户要求，本次仅定位和报告：**没有训练、没有拟合新分类头、没有新增 R18 脚本、没有生成 R18-only 预测。** 这一控制应优先于为了说明互补而追加新的复杂训练机制。
