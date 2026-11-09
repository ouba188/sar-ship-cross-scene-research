# 本轮数据集补缺实现：独立只读审查

日期：2026-11-10。

## 结论

**未发现本次审查范围内会导致错误训练、八类维度不一致、教师索引错位或评价标签进入训练的实质问题；没有需要立即中止OSS1训练的阻断项。**

范围为本目录 `model_init.py`、`engine.py`、`extend.py`、`intervention_inputs.py`，对照 `E:/临时会话/paper_draft_20261108/staged_ablation_20261109/staged.py` 及其 `intervention_inputs.py`。只沿实际调用读取了CRFB、RGRP、M3、原生输入及metrics定义，没有扩展为历史模型审计。

未调用GPU、导入运行实验模块、读取模型权重或大型数组缓存，没有修改代码。读取C盘小型groups.json和OSS1干预complete.json仅用于核实本轮索引。四类初始化/四阶段前向零误差采用主代理已完成验证的结论，没有重复运行。本文是静态接口和小清单审查，不宣称完成八类GPU前向、训练、梯度或峰值显存验证。

## 1. 接口与八类尺寸

- `model_init.py:7` 的四类RegionalGuidance直接委托旧实现；`model_init.py:71` 的四类BaseNetwork同样委托旧构造，不更改原初始化顺序。
- 八类CRFB在 `model_init.py:86` 显式传classes。底层RelationFeedback的reader、逐类transforms、auxiliary及竞争归一化均使用该参数。
- 八类RGRP在 `model_init.py:14` 起使用8响应、9个32维token、288->512投影和7170->8分类头；`model_init.py:35` 的相对响应、uniform展开及后续拼接均匹配。7170和每极化3584是特征布局维度，不应随类别数改动。
- `model_init.py:89` 的aggregate matcher显式传8；底层CompetitiveMatch的one-hot、逐类比较、输出卷积和辅助分数按classes构造。缺少合法同类参考时按可用性掩蔽，不要求每个查询在每个极化都有全部8类合法参考。
- plain/SE分支为4098->8；SE沿用旧两处位置。`engine.py:50` 无CRFB时的零响应、`engine.py:138` hard权重竞争类轴、`engine.py:224` 适配类别计数均已泛化。没有发现活动路径仍将八类截成四类。
- `task.metrics` 使用score的完整类别轴和np.unique(y)计算BA/balanced NLL，dev缺少某类不会索引错位；因此本轮dev BA按实际出现类别平均，评价按其实际类别平均。

## 2. 数据训练与评价边界

- `extend.py:79` 从旧cohorts映射fit/adapt/development/expanded为source_fit/source_adapt/validation/evaluation；参考通过id回查，未把参考追加为重复训练行，也未制造新划分。
- `extend.py:97` 保留本轮已明确的24港dev绝对轨道例外，其余隔离按原定义执行。该已知dev边界不作为新问题重复提出。
- `extend.py:109` 复制冻结的五阶段日程，学习率倍率固定Full/Plain=1、SE=.3。训练阶段cohort依次为source_fit、source_adapt、small、small、small；没有读取selection/evaluation作为拟合或选优样本。
- `engine.py:209` 权重、hard标记、CE及教师监督均按当前训练ids索引。虽然rows包含dev标签，dev仅在阶段末输出指标，不进入优化器损失或阶段/学习率选择。
- 当前前向只将源参考rid的label传给matcher；查询真值不进入匹配前向。`task.legal` 仍以MMSI和平台绝对轨道排除非法query-reference对。
- `extend.py:237` 只加载已完成终点，先与保存的dev分数核对，再对evaluation批量预测；不按evaluation结果挑模型。M3开关结果属于固定权重诊断，不能替代独立删除训练，此口径已在协议中注明。

## 3. GPU前缀子集与全局行号

`extend.py:225` 的last取small和validation最大索引加1，仅切片 `rows[:last]` 和同样前缀的所有输入数组，不重排行号。源参考仍由原始全局索引访问，因此不需要额外remap。

本次读取实际小型groups.json得到：

| 数据集 | source_fit | source_adapt | validation | selection | evaluation | references | 训练驻留前缀 |
|---|---|---|---|---|---|---|---|
| OSS1 | 0–404 | 494–615 | 405–493 | 616–802 | 803–865 | 32条，最小2、最大380 | 0–615，共616条 |
| 24港 | 0–1204 | 1205–1340 | 1341–1500 | 空 | 1501–4575 | 16条，最小57、最大1174 | 0–1500，共1501条 |

两组均没有evaluation落入驻留前缀，也没有reference落在前缀之外。OSS1的small不是连续数组，但orders保留全局ids，readout缓存另建 `lookup={global_id: local_row}`，教师监督也用同一lookup定位；不依赖small连续。

`engine.py:150` 的freeze_delta只按传入的截断rows生成源+dev残差，未按原完整groups强行访问评价行。24港评价有4576行，`extend.py:240/250` 跳过整表resident和freeze_delta，`engine.predict` 每12条读取所需数组及16条源参考；残差由同一固定matcher现场计算。静态路径未发现将训练前缀索引错误用于评价全表的问题。

阶段恢复时，`engine.py:303` 先加载对应阶段state，再configure；经过reference阶段会重新freeze_delta。后续phase2/3冻结matcher且使用固定ImageNet描述符，因此已有delta不会因为CRFB或读出学习而应当自动改变。这与上一轮固定参考描述符的设计一致。

## 4. Teacher cache与冻结前缀

- `extend.setup` 同时重定向engine输出根、intervention_inputs根及prev.records/arrays，prepare/select指向当前数据集，不继续从OSS2读取样本。
- `intervention_inputs.py:20` 依source_adapt原索引顺序记录index/id/start/metadata。每组只存原图之外的变体，位置为start+k-1；原图通过当前原始输入读取。
- `intervention_inputs.py:64` 重复全局idx取得输入，再用局部0..batch-1索引执行临时变体前向。GPU tensor高级索引和原数组读取均产生批量副本，局部替换不会原地改写驻留原图。
- select发生在phase=-1，此时M1/M3尚未启用，因此临时local_a省略desc/mass/write及空rid与实际前向一致；M2和当前载体仍参与自己的教师评分。
- 原图和干预只缓存到layer2最后块conv3之前。此前carrier和intervention阶段只更新layer3末块卷积/相关任务模块，未更新该前缀；SE钩子在该conv3之后，亦不改变已缓存前缀。因此跨arm共享输入成立，教师输出仍逐arm/seed重新计算。
- `engine.py:222` 核对selected索引序列与adapt ids完全一致，随后 `selected[lookup[k]]` 将训练顺序映射回教师记录；同时验证原图教师分数与本模型初始分数一致，不使用别的模型教师分数。
- 已读取OSS1小收据：122组顺序与source_adapt完全一致；各组start连续，metadata非原图条数总和1920与声明count1920一致。未读取early_x.npy/identity.npy内容。审查时24港尚无该完成收据，故不宣称已核实其实际缓存内容；其代码路径与上述索引逻辑一致。

## 收口

没有需要修补的确定缺陷，也未据此添加额外检查或要求重跑四类smoke。八类实际运行与资源是否足够由主代理后续原定准备/训练验证；本审查不占用GPU、不暂停训练、不修改源码。
