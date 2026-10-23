# 冻结协议：additive的sigma / kappa信息分离

2026-10-23。结果产生前冻结。只写本component_isolation目录，不Git、不动父级、不读取测试指标、不进行测试推理。固定18个新头后收口，后续外推由主流程决定。

## 依据与问题

复用24港八类已有效的joint_distribution_readout/additive，不延续OSS2小头搜索。本轮已实际查看既有源验证chemical_disagree 2758及offshore_disagree 2683原VV/VH面板：前者mean救回而旧local未救回；后者旧local为epoch0，仅保留B，不是局部证据识别成功。这些说明不能由单独头或原图外观直接归因完整additive收益。

问题是：在固定B、mu和valid条件下，完整additive收益是否需要kappa所携带的正负不平衡信息，还是sigma的普通离散度已足够？不预定答案，不把统计命名当创新。

## 固定量

- 原3fold，oldseed0/1/2对应seed17/29/43，源身份/日期划分不变。
- 冻结B=T(oldAD)，T强度1，复用原9个score_only校准头，不训练T。训练输入为T(oldAD_OOF)，仍明确T曾用这批源标签拟合，非校准全栈OOF。
- 源train使用原辅助reader的inner OOF mean/sigma/kappa，源validation用对应outer reader。两支OOF组织沿用旧桥接，不把outer训练内矩用于训练。
- 原run_joint.inputs的15槽、源train scaler、valid及两极化不变。复用run_joint.Head('additive')：mean 7->12->1与local 11->12->1，GELU，两分支相加后去类公共分数，名义266参数。
- 每臂相同seed、相同初始化生成顺序和参数张量，末层零初始化严格返回B。完整additive9只加载重放，不再拟合；现有mean_only209是合理信息参照，不宣称容量相同。
- 200步全批次，AdamW lr=.003、weight_decay=.01，jointCE=.5普通CE+.5八类均衡CE；每5步及0步按源validation J=(OA+BA)/2选点，严格大于才更新，平局取最早。沿用原float32训练及float64基线加残差评估。

## 唯一干预

原15槽索引：0=Bcenter；1/2=VV mu/mu_center；3/4=VV sigma/sigma_center；5/6=VV kappa/kappa_center；7/8=VH mu/mu_center；9/10=VH sigma/sigma_center；11/12=VH kappa/kappa_center；13/14=valid VV/VH。

|新臂|标准化后置零槽|保留统计|
|---|---|---|
|sigma_only|5,6,11,12|两极化sigma及跨类中心化sigma|
|kappa_only|3,4,9,10|两极化kappa及跨类中心化kappa|

先用原完整source scaler标准化，再遮槽为0，即该槽恒为源均值；不在原始统计上置零，也不重定义其余槽。名义架构相同但有效可变信息及输入权重自由度较少，不能称统计信息/有效容量完全相等。这是从头训练的信息受限对照，不是删除已训练full分支。

## 数学含义

以原extract_moments.weighted_moments为准，w为归一化footprint权重，r已减空间加权均值及类公共场：sigma=sqrt(sum(w*r^2))，kappa=sum(w*r*abs(r))/max(sigma,1e-12)。kappa是归一化带符号二次矩，不是三阶偏度，也不是只含符号的无幅度量。

每pol、每C4旋转内，E+=sum(w*max(r,0)^2)、E-=sum(w*max(-r,0)^2)，sigma^2=E++E-，sigma>epsilon时kappa*sigma=E+-E-；全零时两统计均0。缓存按四旋转等权平均各统计，平均sigma的平方、平均kappa乘平均sigma通常不等于平均E和/差；本轮不从平均缓存伪造能量。仅复用缓存，不再提图/GPU特征。

## 验证与输出

针对性检查：ID/源OOF覆盖与query身份排除、源train/val分离、原B和full源验证分数重放一次、computed scaler与原full checkpoint相等、两新臂init逐张量相等、未屏蔽槽不变、被屏蔽槽恒0、零头回B、保存选点重放一致。共享NPZ容器只索引源行；不加载baseline的test分数键，不调用自动计算test的父入口。历史CSV先读role/arm列筛源验证，再读取源行，测试分数不进入分析。

保存18选点/曲线/成本及source-only train+validation预测，与原full和mean_only的源预测配对；报告OA/BA/NLL/Brier、逐类/fold/pass/port/VH及独立对象数量。源验证跨fold可能重复，同一seed内记录fold-instance配对，不能把三seed或重复对象当独立船。

每新臂固定seed17、源validation按instance_id再fold取rescued/harmed/CC/WW首例，实际查看原VV/VH及160原尺度crop/pad，不补挑漂亮样本；无某结局则记录缺失。面板列B/full/mean/两新臂分数和真实/最终竞争类分差。选中0步明确为回退。源数据已用于多轮研发，结论为开发诊断，不包装独立外部确认；本轮不bootstrap、不扩臂、不评test。
