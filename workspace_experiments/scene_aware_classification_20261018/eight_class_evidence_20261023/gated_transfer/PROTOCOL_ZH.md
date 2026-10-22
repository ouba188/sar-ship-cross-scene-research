# 八类冻结外推：固定三臂协议

本协议在训练开始前写定。只在 gated_transfer/ 写代码、参数、结果与报告；父目录和 baseline/ 只读。不重提特征、不改标签/有效标记/切分、不执行 Git，不按八类测试调参。使用 karpathy-guidelines 约束实现范围与针对性验证。

## 问题与已有证据

三类实验未证明双区读出独立优于 whole/foreground，所以本轮只使用 whole64，不在八类搜索区域组合。已有八类冻结 layer3 baseline 的测试 OA=39.523609%、BA=40.349061%；这是新 baseline，不是旧 A+D，也不是完整模型胜利的起点证据。

本轮检验：固定原通道方向的正值门控是否能外推到八类；分离共同分数缩放与组间对比能否改变结果；普通类残差头在同一条件输入下表现如何。这里比较的是指定三种小头，不是所有 SE/attention 结构的复现或优劣判定。

## 冻结输入与隔离

- 父15251行队列和全部有效观测保留，三个外层及各自四包源内OOF完全沿用 baseline 缓存。测试4744条、2973身份，不重复计数。
- 每折训练只取 train 下标的 **oof + oq + oof_whole64**。OOF Ridge、q锚和whole条件投影都已在各inner fit重拟合，held过境及MMSI不入该fit。不将 outer source 的 in-sample base/q/whole64 用于训练。
- source validation及test只使用对应outer **base + q + whole64**；模型和训练步数仅按9月source validation选。baseline ridge及其alpha、logit scale=3、原类/身份权重完全冻结。
- whole64范数sqrt32。不拼额外布局/区域/身份/港口/角度特征；保留有效标记用于覆盖分析，不额外作为网络输入。
- q有16个固定原layer3结构组，VV8+VH8，全16组参与。无效VH的原特征块为零，但中心化后的q含缺失状态偏移；不将其清零，否则会破坏A。q不是非负激活，也不是纯物理证据。

## 三臂公式

设冻结分数 A=base=sum_g q_g+b，条件输入 e=whole64。所有 a/m 都逐样本计算。

1. **original_centered**：网络64->32 ReLU->16，a=0.5*tanh(net(e))；score=A+sum_g a_g*q_g=b+sum_g(1+a_g)*q_g。组增益1+a在(0.5,1.5)，但q本身有正负类证据。参数2608。
2. **score_scale_decoupled**：同一网络与a，m=mean_16(a)，c=a-m；score=(1+m)*(A+sum_g c_g*q_g)。共同正缩放1+m在(0.5,1.5)，它本身不改内部对比分数的argmax，但会改概率、NLL和训练梯度。此臂不是简单删去共同项，而是按用户公式重新训练。参数2608。
3. **additive**：网络64->32 ReLU->8，score=A+net(e)。无额外tanh或人为投影，参数2344，少264参数如实记录；不刻意复杂配平。

三臂末层weight/bias全零初始化，初始score精确回到A。这里centered q等价于标准化特征空间上的组门控，不声称中心化本身是原创算子。

## 固定训练与选择

- 三外层 x 三臂 x seeds[17,29,43]，总27头。每臂每seed重新设置torch种子；门控两臂初始化完全同形。
- 全source batch，200步AdamW，lr=.003，weight_decay=.01。使用八类交叉熵与冻结source类/身份权重 w_i=N/(8*类内身份数*身份样本数)，与Ridge使用规则一致。
- 每5步评价source-validation balanced NLL，包含step0。保留比当前最优低超过1e-6的最早checkpoint，未改进则保留step0。无test早停、无额外学习率/幅度/区域搜索。
- 网络训练float32，训练输入只做既有缓存的dtype转换；验证/推理取网络float32输出后，按公式与原float64 base/q计算，以保持冻结A精确不变。保存实际输出并独立重放checkpoint。记录GPU float32公式与显式float64读出的误差，不借精度差换baseline。

## 评价与验证

- 每折、每seed、合并test输出OA、八类BA、普通NLL、balanced NLL、Brier和balanced Brier。Brier定义为每行sum_c(p_c-onehot_c)^2再平均，不除类别数；逐类与采集表保存n和同样的概率/分类统计，子组BA仅对实际出现类别计算并记录类别数。
- 保存逐例base/new八类分数、预测、救回/误伤/CC/WW、原图/观测关联；类和采集收益为配对同一test，不把重复seed当新增独立样本。
- 固定2000次paired cluster bootstrap，分别按MMSI与过境，对三seed均值的OA/BA/NLL/Brier差给95%百分位区间。区间仅作本队列不确定性描述，不据此增删臂或调参。
- 训练前检查zero-init回到A；验证全16组、OOF覆盖和身份/过境隔离。验证共同正比例对argmax的不变性，以及decoupled完整输出与内部对比分数的argmax一致；并明确original共同门控通常不是对含b的A做纯缩放。
- 逐头保存/重载重放，检查预测/分数、step0 fallback、有限性以及baseline不变。只检查能发现实际公式或缓存接入错误的项目，不重复全量提图/审计。

## 图像回看与报告边界

预定主展示original_centered、seed17，按救回/误伤/CC/WW各取obs_id最小的一例；若该臂某状态为空，在固定臂顺序original_centered/score_scale_decoupled/additive及seed17/29/43中取首个可用组合，并如实标注，绝不为凑状态改预测。其余两臂也按相同规则各回看四种状态，重复原图复用，总计最多12个案例事件。

仅复用原PNG、原160 crop/pad、父snapshot mask，不重新跑backbone。VV/VH分别展示实际原输入与**whole有效面积读出mask**；缺失VH明确NOT USED。用保存q、a/m/残差重构真类分差与主要竞争类分差，区别可见图像事实、模型数值变化和未证实机制。不得把fg/bg mask误称为本轮实际控制器读出。

正负结果均保留。八类外推仍在既有开发队列，不是全流程独立盲测；冻结R50、历史PNG拉伸、缺失VH、padding/区域卷积RF混合等边界不因小头涨点而消失。本轮不声称完整模型胜利、纯物理背景因果或双区必要性。
