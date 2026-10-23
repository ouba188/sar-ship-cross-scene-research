# 单一合法源上游划分：可行性与接口

完成更新：下文是启动前的可行性记录。主流程随后明确允许必要的冻结R50源端补提；已按fold0的4702源池完成一次固定四角色划分与三seed完整上游，当前交付见本目录FINDINGS_ZH.md及IMPLEMENTATION_RECEIPT.json。原“尚未执行/待确认”只描述当时阶段，不再是当前阻塞。

2026-10-23。状态：完成有限只读核查；未启动拟合/提取，未改变历史文件。已采用karpathy-guidelines限定实现范围。本次尚未冻结四角色实际成员，尚无可用于协调器训练的新合法预测。

## 结论与阻塞

完整A/D/T及aux的小头可用既有原始描述/GAP重拟合；但新aux W下的双向attention和区域摘要，不能由旧W下的压缩摘要精确恢复。当前“不新提骨干”约束下，不能忠实重建全部策略。不能用旧source_predictions代替，也不能将layer3 GAP预测冒称完整A/D/T。

实际读到的缓存：

|来源|实际缓存|可重拟合的范围|
|---|---|---|
|旧A轴向|same_vessel_gallery_20261019/native_axis_probe_20261020/descriptors.npz|base+axis原始描述，可重新fit标准化与Ridge|
|旧A五点|sampling_preservation_20261020/keep_refs.npy|60×64描述，可重新fit scaler及五点残差|
|旧D|neighborhood_level_split_20261021/features.npz|源标准化、separated残差可重拟合|
|深层aux|eight_class_evidence_20261023/features.npz，layer3[15251,2048]|PCA/标准化/Ridge及线性W可重拟合；没有空间维|
|浅层aux|obb_dynamic_readout/features_source.npz，layer1[15251,512]，computed标记|同上；没有空间维，未计算行不能使用|
|旧attention|attention_source.npz，attention_pol[15,15251,2,8,2]及computed|已按各旧W/旧tau完成非线性空间聚合，不是通道特征图|
|旧区域|summaries_source/fold0_inner0.npz，summary[1339,2,7,8,3]|旧W下mu/h/b；quality/geometry可复用，h/b不能换W后精确重建|

卡点是非线性聚合前的空间信息，不是缺少分类头代码。GAP只保留空间均值；旧类别响应方向和softmax统计不足以计算新方向的softmax统计。新tau也依赖新合法源训练范围，不能把旧tau下h/b直接改温。原实现有意未保存全量F；本次检查的实际缓存未提供所需的未投影空间F。少量病例图不能补全源队列。

也不能直接选一个旧inner W就宣称整链合法：旧A/D/T和attention/body/full头的拟合范围仍可能含未来utility/selection/evaluation；旧摘要还绑定其原选参、tau及特定角色覆盖。需要统一检查并重新限定整个上游的拟合范围。

## 最小忠实路线及待确认边界

建议只有一套外部四角色划分，三seed17/29/43共用，保持原A/D结构、T、layer3 attention和layer1 body/full，不改成新基底。

1. 候选数据只从历史源train/validation索引取得。固定fold0现成源范围为4392 train+310 validation=4702；这是已核实的最小起点，不保证进一步四分后各类覆盖足够。若主流程选择完整源联合范围，应先按索引排除全部三fold历史test并统计覆盖，不能只凭“8750源对象”的总数推定四分可行。
2. 按pass固定seed20261023一次分配四角色，建议初始比例约55% strategy_fit、20% utility_train、12.5% selection、12.5% evaluation；具体比例尚未冻结。跨角色重复MMSI按evaluation→selection→utility_train→strategy_fit优先级排除较早角色中的冲突行，保留排除表。以实际覆盖决定是否可启动，不扫seed挑分类结果。
3. 所有上游模型、标准化、W、tau、模板及上游选点仅使用strategy_fit。该角色内部另分合法训练/选点；现有old held-vessel小头重拟合函数可参考，但不能原样调用其全局fit()，因为它硬编码历史val和输出路径。内部OOF若复用旧协议也必须在strategy_fit内重建。
4. utility_train只学习效用/协调器；selection只选协调器；evaluation仅最后固定推理。一个固定上游对三角色预测即可，不需要为这三个角色分别重训全链。strategy_fit自身的训练内输出不得混入效用学习。
5. 必须额外允许源端一次冻结R50前向，读取layer1/layer3，按新W计算摘要后丢弃大F；这不是训练骨干，但确实超出当前“不新提骨干”的文字边界，尚未执行。全局尺度/区域摘要存在先定tau再提取的阶段依赖，具体一次前向是否能全部完成还需按可保留的小投影缓存确定，不能预先承诺仅一次总遍历或端到端耗时。

若不允许该补提，当前应停在可行性边界，不生成名不副实的完整策略预测。可以另行定义“历史监督表征冻结的下游source pilot”，但那是改变因果/隔离口径，不能作为本任务的自动替代方案。

## 拟采用的数据契约

每seed一个strategies_seed{seed}.npz，三seed统一行序：

- indices: int64[N]，原eight_class_evidence_20261023/cohort.csv位置，不是压缩后本地编号。
- role: str[N]，strategy_fit/utility_train/selection/evaluation。
- logits: float32或float64[N,3,8]，固定策略轴[s0,body,full]；s0是本轮重拟合的完整T(A+D)+two-sided attention，不是单独A或B。
- regional_raw: [N,8,69]，dynamic raw_inputs(s0,info)未标准化，含原共同quality；不能误用已按策略fit标准化的张量。
- 可补strategy_names、class_ids、seed，显式记录轴顺序。

table.csv：row_index、instance_id、obs_id、mmsi、pass_group、port、class_id、role。source_role_counts.csv记录样本/身份/pass/类覆盖；role_exclusions.csv记录排除及原因。主流程协调器只在utility_train拟合其scaler及效用监督。

body/full的策略头各自保留合法strategy_fit scaler；它们共享s0，不能给两条策略分别使用不同的校准基底。原有冻结策略文件永不覆盖。后续receipt应记录每阶段fit/selection索引、选点与类覆盖，明确评价样本不参与任一上游fit/selection；不把reader OOF当成全链OOF。

## 当前交付范围

仅完成缓存能力核查、最小路线和接口说明。未生成新分数，未划分后调参，未读取历史4744目标分数用于选择/训练，未启动GPU或小头拟合，未Git。下一步需主流程确认是否允许必要的冻结空间特征补提，以及采用单fold4702还是源联合范围后，才实施角色覆盖与上游构建。
