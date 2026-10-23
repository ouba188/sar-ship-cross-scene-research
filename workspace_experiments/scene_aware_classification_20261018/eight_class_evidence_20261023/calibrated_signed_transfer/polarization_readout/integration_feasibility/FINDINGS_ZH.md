# 旧完整 A+D 与极化分布读出的桥接可行性

2026-10-23。只读诊断完成；未训练、未提图、未改父目录、未操作 Git。

## 结论

**可以直接进行保留旧完整 A+D 的补充读出实验，无需重拟合旧模型或15个 W，也无需重新提取 sigma/kappa。** 以 `(outer_fold, instance_id)` 连接旧 A+D 源 OOF 与新 moments 源 OOF，两支各自使用自己的 held 分配。三折13,229条源训练行暴露均可连接；两支 query MMSI 拟合/源选模暴露均为0，outer-test身份暴露均为0。

纠正初步判断：旧 inner3、新 inner4 不同，**不意味着不能合法拼接独立交叉拟合分支**。不同 fit 集合无需逐项一致。不能做的是按相同 inner 编号取 W，或把 outer W 的 moments 当作源 OOF。这里的结论来自逐行归属及实际 fit/held 身份核对，不是仅看文件名。

这可以称为“冻结两支上游、身份 OOF 训练补充读出、外层验证测试隔离”。不能称为：整条链过境 OOF、校准 C 的二级 OOF，或同一份最终全源模型从未接触校准船的独立留船校准。后者是更强且不同的设定，不是当前普通源监督学习必须满足的前提。

## 实际匹配数量

| 外折 | 源训练 old/new/匹配 | 验证 old/new/匹配 | 测试 old/new/匹配 | 两支 query 身份暴露 | 两支 outer-test 身份暴露 |
|---|---:|---:|---:|---:|---:|
| 0 | 4392/4392/4392 | 310/310/310 | 1561/1561/1561 | 0 | 0 |
| 1 | 4419/4419/4419 | 286/286/286 | 1632/1632/1632 | 0 | 0 |
| 2 | 4418/4418/4418 | 319/319/319 | 1551/1551/1551 | 0 | 0 |

- 旧 cohort 15,252行，新 cohort 15,251行；唯一旧有新无记录为已排除的 obs_id=31237。该记录不在旧有效训练/验证/测试中。
- 15,251条共同记录的 instance_id、MMSI、class_id、外 fold、product、time、VV/VH原图路径完全一致，所有差异计数为0。
- 删除记录后 **3,103行的位置索引发生偏移**。不能直接把旧 `rows` 用于新 moments；必须经 obs_id/instance_id 映射。`row_mapping.csv` 已保存完整映射。
- 三折各角色的观测顺序也一致，但原始整数行号不一致。源13,229和验证915是跨折暴露次数，不能当独立样本数；测试共4,744条。
- 旧9个 inner、新12个 inner 的 fit/held 都覆盖8类。旧 held 与 fit 共享175–186个过境组，但无共享MMSI；新支两者的MMSI和过境均不相交。因此组合继承身份 OOF，不能升级为全链过境 OOF。
- 验证有效VH为274/265/294，测试为1458/1529/1449；测试308条缺VH仍保留，不引入额外筛选。
- 36种 old-inner/new-inner 配对中没有相同 held 集合；这个数字只说明不能互换归属，不否定独立分支 OOF 的合法性。

## 可复用的真实文件

以下路径前缀 `OLD=E:/临时会话/same_vessel_gallery_20261019`，`EIGHT=E:/临时会话/scene_aware_classification_20261018/eight_class_evidence_20261023`。

| 内容 | 实际文件与读取方式 | 数量/形状 |
|---|---|---|
| 旧完整模型全队列分数 | `OLD/geometry_evidence_gate_20261021/inputs_f{0..2}_s{0..2}.npz`，`base + residual` | 9文件，各15252×8 |
| 旧完整模型源 OOF | `OLD/held_vessel_gate_20261022/oof_f{0..2}.npz`，`base[old_seed] + residual[old_seed]`；行是文件中 `rows`，归属为 `inner` | 3文件，各3×源训练数×8 |
| 旧测试精确参考 | `OLD/neighborhood_level_split_20261021/scores_f{fold}_separated_s{seed}.npz`，`scores` 已是 A+D | 9文件，全队列缓存映射到test后最大误差0 |
| 新极化分布 | `EIGHT/calibrated_signed_transfer/signed_features.npz`，`moments_pol`，按 `reader_names` 选outer或各自inner | 15×15251×2×8×2；末轴sigma/kappa |
| 新分组/实际fit | `EIGHT/baseline/fold{fold}_inputs.npz` 与 `models/fold{fold}_inner{0..3}.npz` 的 `fit/held` | 3outer、12inner |
| 旧分组 | `OLD/held_vessel_gate_20261022/inner_partitions.csv` 和各 `oof_f{fold}.npz` | 两者行与归属完全一致 |

`score_inventory.csv` 列出上述21个旧分数文件的绝对路径、字段形状与大小。旧 `held_vessel_gate` 目录中的各种 `scores_*` 是后续门控结果，**不是本次要保持的裸 A+D**。也不能把 `inputs_*` 中 `content` 当类别分数。

旧 A+D 三seed复核 OA/BA：41.357504/42.688248、41.968803/43.121502、41.209949/42.312530。只是验证冻结入口，未用这些测试数字选方案。旧seed索引为0/1/2，新读出实验常用17/29/43；后续需在协议中固定配对并同时保存两个字段，不能混淆为同一随机初始化。

## 最小接入及成本

1. 训练输入：旧 `oof_f{fold}` 的 A+D，与该 instance 在新支自身 held 对应 W 下的每pol moments 配对。推理/源验证输入：旧全源 A+D 与新 outer W moments。两支均按既有源划分，查询标签没有进入该行上游拟合；源码中的选模只使用共同外层源验证，不使用outer test。
2. 如沿用已验证的“先校准再补充”组织，需**重新训练旧 A+D 专属 C**，冻结后再训练补充头。不能迁移当前以新layer3 A为输入训练出的 C 或极化头。新的descriptor用旧模型经C所得B替换分数项，其余真实pol moments保持不变；标准化仍只用当前源OOF训练材料。
3. 3fold×3seed时，C为9条轨迹，每个预先指定补充臂再9条。此处没有选择臂、启动训练或追加网格。所有上游重拟合数=0，新增原图读取/提取数=0；只需小数组连接、源统计及小头训练。
4. 成本依据：既有18个standard-controls小头总运行18.29秒、训练及重放15.47秒；可据此预期新9C+单臂9头是秒到几十秒量级的缓存实验，不保证精确时长。旧54内层神经头+27ridge候选+36门控历史共50.5秒、新15W历史28.39秒、moments提取219.39秒，**均无需本次重做**。

## 必须保留的语义边界

- 新 moments 的 W 是新layer3线性分类器方向，不是旧完整 A+D 的权重或Jacobian。旧内部ridge权重为8×796，后面还叠加非线性 keep_refs 与邻域D；新 W 为8×2048。合法的实验解释是“用独立冻结类语义分布证据补充旧完整模型”，不是“严格分解旧模型自己的局部证据”。不需要为了此目的强行转换 W。
- 旧与新 OOF 都排除 query 身份，但小样本fit、尺度及旧D选零步导致的训练/推理差异仍可能存在。这是泛化边界，不是由inner数量不同自动产生的泄漏；不得为消除差异而把全源in-sample分数喂训练。
- C如果用本批旧OOF分数及标签训练，再将C(OOF)提供给补充头，则是普通两阶段源监督；C不是二级OOF，不能称补充头学到完全样本外校准错误。外层验证测试仍合法隔离，公平对照需共享相同C与这一限制。
- 若“独立校准”特指校准船从最终冻结上游的训练、预处理及选步中全部排除，则现有源train已被最终上游拟合，源val已参与上游选择，**现成train/val中没有可直接宣称满足该定义的校准集**。需要另外明确留船/留出数据并重做受影响上游，属于不同设定。本次不扩大到该设定，也不以此否定当前可行桥接。
- 历史 `held_vessel_gate` 报告明确未接完整目标支持适配。本报告“旧完整 A+D”只指已定位的冻结A加邻域D链，不把它改称包含所有历史适配阶段的完整系统。

## 产物与验证范围

`bridge_counts.csv`、`identity_exposure_counts.csv` 为核心决策表；另有 `outer_partition_counts.csv`、`inner_coverage.csv`、`source_inner_membership.csv`、`row_mapping.csv`、`inner_crosswalk_counts.csv`、`old_score_metrics.csv`、`score_inventory.csv` 和 `summary.json`。

`diagnose_bridge.py` 已实际运行完成，检查分数有限、既有分数重放、标识对齐、保存fit/held的身份隔离和源选模/测试边界。旧fit归属依据保存的inner分配与实际训练源码，新fit归属直接依据模型npz；没有重训来复证训练历史，也未做全库审计。无新增实验，故未重复原图回看。无运行中的诊断进程。
