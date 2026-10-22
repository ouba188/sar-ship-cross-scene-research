# 八类冻结 layer3 baseline 与合法 OOF

## 范围与来源

只写本 baseline 子目录。父 cohort.csv/features.npz 保持不变；15251 条合法图像各保留一次，obs_id=31237 已由父任务排除，不再筛样、补身份或改标签。原始 160 crop/pad、无 resize 的冻结 R50 特征直接复用，不重新提取。该结果是**新的 layer3 baseline**，不是旧 A+D，也不能与三类 strong reader 数字混为同一实验。不训练 attention，不运行 Git；模型和缓存仅本地。

采用 karpathy-guidelines 的局部实现与针对性验证原则。唯一需要拟合的是指定 baseline、其 OOF 重拟合以及无标签条件投影。

## 外层与内层

- 原样调用 same_vessel_gallery_20261019/conditional_evidence_pilot/run_pilot.py 的 indices(f,fold)，fold=0,1,2。其身份选择种子保持原函数的 20261019+fold，不改为新种子。
- source 日期 <2025-09-01；validation 为 2025 年 9 月；test >=2025-10-01。train/validation/test 的 MMSI、product 和过境互不相交。
- 每个外层仅对其 source 过境排序，以 RNG 20261023 置乱后 roundrobin 分成四包。过境键为 Sentinel 产品名中的 platform_absoluteOrbit_datatake，包含平台防止跨卫星轨道号碰撞；不是单个切片 product。
- 每个 inner held 包从 source 选取，fit 排除该包全部过境和该包所有 MMSI。held 覆盖每条 source 恰好一次，非 source 的 OOF 位置为零且 oof_valid=false。每个 inner 单独拟合预处理、Ridge 和条件投影。
- 所有 split 的逐类数量落盘；实际缺类则报告并停止，不合并包、不换种子、不补类别。本次实际 12 个 inner fit 均含八类。
- alpha 复用各外层 validation 的选择。OOF held 标签不参与相应 inner 拟合；外层验证集参与 alpha 选择，这是指定的 source-validation 协议，不是再嵌套 alpha 搜索。

## 八类读出

沿用 run_strong.View：StandardScaler -> PCA128(randomized, random_state=17, iterated_power=4) -> StandardScaler。所有步骤仅 source fit，layer3 原通道为 VV1024 + VH1024。

Ridge 使用显式八类 onehot，按 source 类/身份加权：w_i=N/(8 * 该类身份数 * 该身份样本数)。每类权重总和 N/8。候选 alpha=[1,10,100]，固定 logit scale=3，以 source-validation balanced NLL 最小选择，完全不按测试结果选择。NLL 按八类分别平均后等权，OA/BA 为百分数。

保持 float32 特征的原 View/head 拟合，评价在 float64 中显式执行同一线性映射。alpha 候选的 validation NLL 也使用该评价精度。记录 float32 原管线与 float64 评价的分数误差及 top1 差异，不偷偷替换已有 baseline。

W 为含 logit scale 的 8x2048 原通道权重；主锚 h0 是 fit 原特征无权均值，另存 h0_weighted。q 按原通道 16 组，每个 pol 八组、每组连续 128 通道；不按 PCA 轴分组。b=score(h0)，base=sum_g(q_g)+b。加权锚的 b_weighted 另存；intercept=b-W@h0。inner 独立 W/h0/b 保存在 models，oq/ob 按 held 行装配，因此不能用 outer b 重构 OOF。

## 条件投影

只用各 fit 的 whole stem256，无标签拟合 StandardScaler -> PCA64(full) -> post StandardScaler，遵循父 Baseline.project 的标准化链。whole/fg/bg 全部进入同一坐标系；前32维供 fg/bg 双区，不另拟合区域 PCA。

- whole64、fg64：投影后 L2 归一，再乘 sqrt32。
- dual64：fg 前32、bg 前32各 L2 归一后乘 sqrt32，再拼接除 sqrt2。双区均有效时总范数 sqrt32；单区无效时保留另一半，不人为补足范数。
- outer 条件投影用于全队列；oof_whole64/oof_fg64/oof_dual64 用各 inner fit 的投影，只填 held 行。没有套用 outer PCA 到 inner held。
- 父缓存已将无效 pol 的原特征块置零，检查并原样保留；valid(N,2)、regional_valid(N,2,2) 均随缓存保存。区域任一 pol 有效即保留投影，只有双 pol 均无效才清零整个区域向量。不以 all-pol 判定抹掉有效 VV。
- PCA 混合极化，缺失 VH 的零原块在中心化之后不必仍是零坐标；这不是补出 VH。后续模型须读取缺失标记，不能把投影看作真实双极化完整观测。

## 输出约定与边界

fold{0,1,2}_inputs.npz 的全部 N 维数组均按父 cohort 顺序；train/validation/test 是该顺序的整数下标。q/oq 为 N x16x8，base/oof 为 N x8，ob 为逐行 OOF 锚偏置。models 保存 3 outer+12 inner 的线性参数和条件投影参数，未导出训练网络。

predictions.csv 包含各外层 validation 和 test，读取 pooled 时必须筛 role=test；OOF 单列 oof_predictions.csv。按原 indices，不是15251条都属于 test：时间/身份角色限制保持原样，全队列只用于缓存对齐。

原图回看为两例正确、两例错误且四种真类，固定类别顺序 bulk/container 正确、product-chemical/general 错误，每组按 obs_id 取第一例。未比较两个分类器，因此没有“救回/误伤”；不能把单一 baseline 判对/错冒称 CC/WW 翻转。展示原160输入和实际 pooling mask，VV/VH分开，无效VH明确标识；复用父图作八类覆盖补充。可见外观不能单独证明物理机制；深层卷积 RF 跨越区域与 padding 的边界仍然存在。
