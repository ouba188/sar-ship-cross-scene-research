# Pipeline 独立核对记录

日期：2026-10-14。范围：代码、既有完成记录、NPZ数组头与缓存路径的只读核对。本支线没有训练、重拟合、重新预测，也没有修改 model/run/analyze/DESIGN。用户要求收敛时 verify.py 尚未创建，因此不追加验证脚本或大批检查。下述“确认”区分代码/代数确认与主线已经完成的数值记录；本支线没有独立重跑主线数值验证。

## 1. 可立即用于方法文档的结论

1. **旧2752维 compact/class-reference 候选仍有 PCA 丢弃空间补充项。** 不能写“没有额外PCA补充模块”。没有分频基础头，不等于没有PCA余量核。
2. **真正 lean192 已在独立入口明确删除余量坐标。** `run_lean.py` 的 `KEEP=np.r_[0:128,2176:2240]` 与 `transform(z)=z[:,KEEP]*sqrt(2)` 正确，得到标准化 R50-128 + SSL-64 主坐标 `/sqrt(192)`。这是完整 cached-feature 双头候选，不是已经正式替换的旧 full。
3. 删除余量时可以保留 C4 基础特征、基础源先验、基础支持适配，以及192维专家源分类头及其 `pd`。**无须重训骨干或重拟合这两个源分类头**；新核需要重新求解专家支持残差系数，基础支持系数可复用。
4. lean192 的类别中心、8-cluster中心、共同带宽、响应源中心和能量尺度需要在192维源坐标重建。当前 `run_lean.py::lean_banks` 已这样组织；原2752维 bank 不能直接截列并沿用旧校准。
5. 数值归一化修正的数学方向正确，且与PCA剪枝是两个不同改动。原始、归一化修正、lean192应分别保留和命名。主线的历史复现通过不能证明旧2752维候选已经剪枝。

本报告读取 `run_lean.py` 时，lean192仍在运行。随后主代理通知已完成：OSS2跨采集B64类别参照相对自身原核增量由约+0.976降至-0.027个百分点，完整class BA由31.733降至30.163；F对应增量仍约+0.505，但绝对BA下降约0.518。此处仅记录主代理回传，未独立重算；完成实验不等于正式替换旧full。

lean同时改变原支持核与参照几何，性能损失不能全归因于参照或原核。主线已提出仅B64缓存的2x2交叉读出，区分2752/192原核与2752/192参照的作用位置；本支线不重复执行，也不提前填写其结论。

## 2. 2752维的实际组成

定义 `x_R` 为源归一化 R50 PCA128 主坐标，`x_T` 为源归一化 SSL PCA64 主坐标；`d_R/d_T` 是去除对应PCA重建后、按源统计标准化的余量，分别以原2048/512维空间存储。

```text
r = sqrt(0.5) * concat(x_R / sqrt(192), d_R / sqrt(2560))  # 2176维
t = sqrt(0.5) * concat(x_T / sqrt(192), d_T / sqrt(2560))  # 576维
z = concat(r, t)                                         # 2752维

z_lean = sqrt(2) * concat(r[:, :128], t[:, :64])
       = concat(x_R, x_T) / sqrt(192)                     # 192维
```

`r` 不是“raw2048 + discarded128”；第一块是128维主坐标，第二块是2048维余量。补充项也不是另一次PCA的2560个独立主成分，而是丢弃空间投影余量的原空间坐标。

具体证据：

- [backbone_ablation_core.py:54](E:/Hermes/paper_tracking/sensitivity/oof_reliability_check/backbone_ablation_core.py:54)，`r50_views`：构造丢弃空间余量，返回 retained `/sqrt(192)`、residual `/sqrt(2560)`。
- [projection_residual_expert.py:11](E:/Hermes/paper_tracking/sensitivity/oof_reliability_check/projection_residual_expert.py:11)，`discarded_coordinates`：`(x-pca.mean_) - ((x-pca.mean_) @ components.T) @ components`。
- [dual_cue_decomposition/run.py:41](E:/临时会话/dual_cue_decomposition_20261012/run.py:41)，`ssl_map`：64维主坐标、512维余量，各自缩放后整体乘 `sqrt(.5)`。
- [study.py:133](E:/临时会话/port_shift_matched_acquisition_20261013/study.py:133)，`encode`：R50的两块拼接后乘 `sqrt(.5)`；基础头 `b` 另走C4 PCA128路径。
- [class_reference_external/experiment.py:106](E:/临时会话/class_reference_external_20261014/experiment.py:106)，`encoded`：直接拼接完整 `r/t`。当前主线 `run.py::prepare_external/prepare_ports` 沿用该表示。

本支线实际读取过NPZ数组头：外部缓存 F 有7594个已编码行，OSS2有2528行；两者 `r` 均2176维、`t` 均576维、`b` 均128维，`pb/pr/pd` 均8维。当前主线 `BANK_FITS.json` 也记录 external/ports 的输入维数为2752。上述维数并非仅从变量名推测。

**缩放解释：** 单纯删列会留下 `0.5 * K_retained + 1`；乘 `sqrt(2)` 后则为 `K_retained + 1`。主线选择的是去掉补充混合分支后恢复完整主坐标权重，合理且定义清楚。但2752到192的绝对差异同时包含余量删除和主坐标核权重恢复，不能声称它是“其余核权重完全不变”的单项消融。类别与3个聚类对照均使用同一个lean表示，彼此的匹配比较不受此表述问题影响。

## 3. 最小完整推理定义与复用边界

| 环节 | 当前compact实际实现 | lean192保留/变动 |
|---|---|---|
| 图像表示 | R50 raw2048、R50 C4 pooled2048、SAR-SSL R18 extra512缓存 | 复用同一批冻结缓存；不能把C4说成R18也已做C4 |
| 源变换 | R50 PCA128、SSL PCA64及源中心/尺度；另有余量统计 | 主坐标PCA仍保留；余量不再进入专家核 |
| 基础源先验 | C4投影的128维坐标上 balanced RidgeClassifier，alpha=1 | 不变，缓存 `pb` 可用 |
| 专家源先验 | 标准化 R50-128 + SSL-64 上 balanced RidgeClassifier，alpha=1 | 不变，缓存 `pd` 可用；该分类头本来就不是2752维输入 |
| 基础支持适配 | `Kb=bq@bs.T+1`，对源先验残差做类平衡ridge8 | 同支持集时特征、系数、概率均可复用 |
| 专家支持适配 | `K0=zq@zs.T+1`；类别/cluster为 `.75*K0+.25*Rq@Rs.T` | 改192维z并重建R，重求解专家支持系数 |
| 输出 | 基础与专家分别softmax，再 `.5*base+.5*expert` | 不变，不能改称平均logit或单一联合核头 |

源拟合入口是 [evaluate.py:27](E:/临时会话/public_oss_comparison_20261012/evaluate.py:27) 的 `prepare`：PCA在允许源样本拟合，基础head在C4坐标拟合，dual head在192维坐标拟合。完整候选只需 base/dual 两个head，第三个r50 head是已有诊断对照，不是部署必要组件。

基础 `b=(x4-residual_center)/residual_scale/sqrt(128)`；基础 prior用head作用于 `x4`，不是作用于 `b`。专家 prior用 [compact_fusion_source.py:32](E:/Hermes/paper_tracking/sensitivity/oof_reliability_check/compact_fusion_source.py:32) 的 `_expert_inputs` 所给192维坐标；该函数没有丢弃空间补充项。

共享残差求解器是 [unified_head_core.py:11](E:/Hermes/paper_tracking/sensitivity/oof_reliability_check/unified_head_core.py:11) 的 `class_weights/coefficients`。对于支持标签中实际出现的类别，`w_i=B/(C_present*n_yi)`，目标为8维 `2*onehot(y)-1`，不是普通0/1回归目标：

```text
A = W^(1/2) [W^(1/2) Kss W^(1/2) + 8 I]^-1 W^(1/2) (Ypm - prior_s)
scores_q = prior_q + Kqs A
```

缺失支持类仍保留在8类输出中。`bank=None,rho=.25` 是本次明确设计的shrink对照，不是“缺bank时漏报错误”。新类 [model.py](E:/临时会话/class_reference_core_20261014/model.py) 的 `fit_support/predict` 与上述双头公式一致；每个arm重求解系数，不应将旧专家系数直接搬到新核作为正式预测。

## 4. compact与冻结旧full的具体差别

冻结口径见 [FREEZE_AND_CLAIMS_ZH.md](E:/临时会话/paper_core_freeze_20261013/FREEZE_AND_CLAIMS_ZH.md) 第2、5、6节：冻结的是25任务既有 `dual_refit_late` 资产与协议，当时没有完成架构剪枝。

| 项目 | 冻结旧full | external_scene_transfer compact |
|---|---|---|
| 基础核 | phi核 + 条件RBF加权mean核 + `.25`条件加权selective分频embedding核 | C4基础128维线性核加常数1，无mean条件核/分频项 |
| 专家核 | retained/complement各0.5；港口还含stable及入射角项，并涉及drift变换 | 仍有retained/complement各0.5，但没有港口stable/drift/incidence项 |
| 专家表示来源 | 港口matched_joint、外部joint-tail路径及其任务几何 | frozen原始/e206缓存与新共同源几何；不是同一个任务tail表示的简单删除 |
| 源先验 | 冻结任务各自source集合/几何，dual-refit专家先验；基础head继承旧路径 | external允许源交集28854行，共同新PCA/尺度和3个ridge源头 |
| 支持/查询协议 | 原25任务B64五抽样；旧目标行全集/对应掩码 | 已知一致身份船只选定查询、B16/64；外部全局及采集匹配任务 |
| 正式地位 | 2026-10-13冻结参考 | 独立诊断/研发候选，不能继承旧full分数 |

旧full可运行入口定位：[backbone_ablation_run.py:49](E:/Hermes/paper_tracking/sensitivity/oof_reliability_check/backbone_ablation_run.py:49) `predict_task`，其中 `dual_refit` 使用重新拟合专家先验、支持适配、`late`概率融合。基础核见 [unified_head_run.py:29](E:/Hermes/paper_tracking/sensitivity/oof_reliability_check/unified_head_run.py:29) `base_kernel` 与 [frequency_band_core.py:20](E:/Hermes/paper_tracking/sensitivity/oof_reliability_check/frequency_band_core.py:20) `old_kernel`；专家港口/外部分支见 [backbone_ablation_core.py:136](E:/Hermes/paper_tracking/sensitivity/oof_reliability_check/backbone_ablation_core.py:136) `AblationContext.build`。

这些差异说明：当前compact不是旧full的纯“去分频”消融，lean192也不是旧冻结25任务主表的一次已验证原位剪枝。

## 5. 源bank与8-cluster公平性：已确认的静态接入

- 外部：`run.py::prepare_external` 从旧 `deployment_memory.pkl['source_rows']` 取得13755个源船只观测，并在同一外部几何上重新编码。旧 [experiment.py:114](E:/临时会话/class_reference_external_20261014/experiment.py:114) `memory` 先限于允许源行，再排除未知/冲突身份，固定seed每船抽一条。28854行源head几何允许包含183条未知身份行；它们不进入类别/cluster参照bank。这两种样本数差异有明确来源，不是泄漏证据。
- 港口：`run.py::prepare_ports` 读取pilot的10633个源行，按 `np.unique(ships)` 顺序、seed20261013每船抽一条，得到6047条。它与 [pilot/core.py:10](E:/临时会话/scene_applicability_design_20261013/pilot/core.py:10) `build_memories` 和 [repair_relative.py:13](E:/临时会话/scene_applicability_design_20261013/pilot/repair_relative.py:13) 的chosen机制一致。pilot入口已先筛已知、一致标签身份；新chosen本身不再次clean并不构成已发现错误。
- 港口的 `old.source_metadata()` 读取的是共同24港源manifest；`identity=mmsi`，并非把F/OSS2目标metadata当港口源。港口源行被映射到pilot encoded的manifest全局行，代码有行号相等断言。
- `run.py::banks` 把同一 `source_z` 同时用于class中心一致性断言和3个 `cluster_bank`。聚类调用是 `KMeans(...).fit(z)`；labels只用于事后cluster类别构成统计，不进入KMeans。3个预定seed各 `n_init=5`，都报告；cluster平均是三次拟合性能的平均，不是预测集成。
- 每域使用共同带宽、rho=.25、ridge8及融合.5。类别与cluster分别由相同源观测计算自身响应中心/能量尺度，这符合相同处理规则；不是要求两个不同参照映射有相同数值尺度。
- 类别中心按类组织，而无标签KMeans受源类别比例影响。因此若类别中心更好，可以支持“类别组织的参照有价值”，不能单凭这组对照称抽象语义的唯一作用，也不能把源标签监督预算说成完全相同。

截至只读核对，没有发现这条调用链将目标查询标签传给bank拟合。此判断来自实际代码流及历史源隔离协议；本支线没有新增独立元数据全量join或重新拟合聚类。主线原 `banks` 自带的中心一致性断言不再重复执行。

## 6. 归一化修正：数学确认与阈值边界

令 `a_c=-||z-mu_c||^2/tau2`，`v=exp(a)`，`n=||v||_2`。旧实现输出 `v/max(n,eps)`，eps=1e-15；修正实现先减 `max(a)` 再exp并作L2归一化。

- 若 `n>=eps`，两个表达式在实数数学中相同，实际浮点只应有舍入差异。
- 若 `0<n<eps`，旧输出等于正确单位向量乘 `n/eps`，其模长小于1；当norm计算下溢为0时旧输出更失真。故旧floor不只是无影响的数值保护，它改变了远离bank样本的表示。
- `np.linalg.norm(exp(a))==0` 不必意味着每个 `exp(a_c)` 都已为0：平方求和也可能先下溢。已有诊断的 `all_zero` 字段实际检查的是norm是否为0，应解释为“计算所得范数为0”，不据此断言所有指数分量均为0。
- 新实现在有限距离、正带宽条件下，减去行最大值保证至少一个指数项为1，范数处于 `[1,sqrt(8)]`，无需epsilon floor。很小的非最大分量仍可能下溢，但不会再把整行压成非单位向量。

令 `h_c=(2*z dot mu_c-||mu_c||^2)/tau2`，因 `a_c=h_c-||z||^2/tau2`，共同径向项严格消去：

```text
unit_c = exp(h_c) / sqrt(sum_j exp(2*h_j))
       = sqrt(softmax(2*h)_c)
unit_q dot unit_s = sum_c sqrt(p_qc * p_sc)
```

这个解释适用于归一化后的原始8维响应；**随后源中心化/能量匹配后的R，其dot不再直接等于上式的概率分布亲和系数**。不能称p为已校准真实类别后验，也不把这个代数等式作为新数学发明。旧floor恰破坏共同径向项消去和单位向量性质。

当前 [model.py](E:/临时会话/class_reference_core_20261014/model.py) `unit_response/reference_map` 实现上述修正；[repair_normalization.py:20](E:/临时会话/class_reference_core_20261014/repair_normalization.py:20) `recalibrate` 遍历class与全部3个cluster，保持各中心和带宽，重新计算各自源中心/能量尺度；重新求解支持适配，基础分支不变。未发现只修类别对照而漏修cluster的接入。

主线 `reference_activity.csv` 中class_mean报告：F查询552/4812、支持329/2582触floor；港口支持1，OSS2查询1/支持2。这里是已读取的主线诊断证据，本支线没有独立重算这些数。`relation`与`refit`出现大幅反向抵消是促使检查核尺度的线索，本身不充分证明所有涨跌的根因，也不保证修复后的分类收益必然为正。

## 7. 可复用入口和缓存

| 用途 | 文件/入口 | 复用边界 |
|---|---|---|
| 外部共同源几何/head | `E:/临时会话/external_scene_transfer_20261013/fits/geometry.pkl` | 本支线确认存在；28854源行，不是旧full任务几何 |
| 外部编码特征 | 同目录 `f0921_encoded.npz`、`oss2_encoded.npz` | 已确认r/t/b/pr/pd/pb头；rows是所用子集，须按rows映射，不能当完整manifest位置 |
| 外部任务与预测 | `E:/临时会话/class_reference_external_20261014/EXTERNAL_ENROLLMENT.json`、`predictions/episode_*.npz` | 24任务220支持单元，保留query/support身份与固定.5融合参考 |
| 外部源参照 | 同目录 `deployment_memory.pkl` | 原2752坐标bank及13755 source_rows；lean仅复用源行选择，不复用旧校准 |
| 港口源与旧校准 | `E:/临时会话/scene_applicability_design_20261013/pilot/encoded.npz`、`source_rows.npy`、`memories.pkl`、`relative_repair/PROTOCOL.json` | 港口源几何与外部不同，不能交换bank |
| 港口48场景 | `E:/临时会话/scene_applicability_expansion_20261014/encoded.npz`、`ENROLLMENT.json`、`predictions/episode_*_b64.npz` | 原6/扩展42身份、支持及同/异采集任务可复用 |
| 当前原2752候选 | `E:/临时会话/class_reference_core_20261014/run.py`、`legacy_model.py` | 保留原floor结果和历史复现 |
| 归一化修正2752 | 同目录 `repair_normalization.py` -> `normalization_repair/` | 同表示/中心/带宽的单次数值修复；不是剪枝 |
| 真正192维候选 | 同目录 `run_lean.py` -> `lean192/` | 主坐标权重恢复、新参照bank与支持系数；不要求精确复现原2752预测 |

低层原始缓存路径见 [external_scene_transfer/run.py:213](E:/临时会话/external_scene_transfer_20261013/run.py:213) `arrays_for` 和 [class_reference_external/experiment.py:100](E:/临时会话/class_reference_external_20261014/experiment.py:100) `source_arrays`；本支线检查过这些主要文件存在。`external_scene_transfer/run.py predict`、`evaluate.prepare`、`experiment.py external/validation` 都可能写缓存或重新拟合，不是本次只读审阅命令，本支线未运行它们。

## 8. 已验证边界及收尾判断

主线完成记录报告700支持单元、4200专家求解、1400原始/class历史复现，最大差异1.3322676295501878e-15，并有18个独立KRR检查；这些证明原2752新封装接入与旧compact一致，不能证明PCA已删、旧floor数学正确或旧full已替换。

本支线确认了实际组件/维数、源bank静态传参、双头公式、修正的代数等价边界以及lean变体复用条件。发现的实质问题是原“无PCA补充项”描述与实际2752输入冲突，主代理已明确纠正并另跑lean192；此外应准确解释诊断 `all_zero` 为范数零，以及区分中心化前后的概率响应亲和解释。未发现需要本支线平行修改共享模型脚本的其他已证实接入错误。

用户要求停止扩展后，不增加verify.py、不重复已有KRR/逐cell复现/宏BA重构，也不把未执行的独立数值检查写成已通过。最终lean结果和主线结论以其完成记录及报告为准。
