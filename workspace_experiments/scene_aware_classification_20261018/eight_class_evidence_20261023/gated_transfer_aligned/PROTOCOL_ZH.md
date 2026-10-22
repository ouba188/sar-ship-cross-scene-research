# 条件坐标语义纠错：固定三臂重放

本协议在修正训练前写定。原 gated_transfer/ 的27头、预测、指标、12个原图事件及代码保持不变；本目录独立保存诊断、修正27头及原/修正配对报告。不重训Ridge/PCA、不改父特征/缓存/标签/划分、不Git，不增加超参数。

## 实际问题与先行诊断

原baseline每个inner的conditional分别拟合StandardScaler/PCA64/postscale，oof_whole64按held行拼接；单个共享MLP却在推理时接收outer PCA坐标。不同fit的PCA轴没有符号、排列或旋转对齐，因此同一索引不保证同一语义。q已回到固定原layer3通道，不存在这个PCA轴问题；本轮不改q。

先对每折四个inner的同一source held行量化：oof_whole64与whole64逐样本cosine；64个同索引维度Pearson相关；inner/outer PCA同索引轴signed/absolute cosine，以及所有outer轴中absolute cosine最优的匹配索引。另列含各自StandardScaler尺度的有效原stem轴比较，避免将标准化坐标的余弦误作完全相同原通道映射。相关零方差维度标记undefined，不伪填相关。

这些数据可证明共享MLP面对的坐标不一致，不证明原全部损失由它造成。原负结果保留，但不能单独据其判定门控方向无效。

## 唯一训练改动

原训练输入：oof[train]、oq[train]、oof_whole64[train]。

修正训练输入：**oof[train]、oq[train]、whole64[train]**。

推理保持base/q/whole64。whole64预处理只拟合outer source train的无标签whole stem，validation/test未入fit；接受源内共享的无标签预处理。这是**基线读出/贡献OOF，条件预处理outer-source共享**，不是整流水线OOF。不能声称held样本完全不参与条件预处理；它们确实参与outer source的无标签PCA拟合。

其余严格复制原protocol：original_centered与score_scale_decoupled的64->32ReLU->16、additive64->32ReLU->8；2608/2608/2344参数；16组全有效；末层零初始化；seeds17/29/43；200步全batch AdamW .003/.01；八类/身份权重不变；每5步含0按sourceval balancedNLL选择、改善阈值1e-6。原公式、精度路径、baseline、alpha、训练队列均不变。门控公式见原gated_transfer/PROTOCOL_ZH.md，不重新选择范围。

## 评价

先报告原输入坐标诊断，再报告修正三臂的OA/BA/NLL/Brier及逐折/seed/类/过境、有效VH分层。对同一arm/seed/instance做原vs修正的精确配对差，固定2000次MMSI/过境cluster bootstrap，seed先按同一实例平均，不能把seed当额外独立样本。

沿用原zero-init、正共同缩放argmax、checkpoint重放和固定baseline验证。回看修正的三臂救回/误伤/CC/WW各一例，采样仍按seed17优先、obs_id最小、固定后备顺序。原12个事件已经实际看图且保留；修正新图和同案例跨原/修正分数全部写本目录。只读原PNG和160crop/pad，不重提特征。原图展示实际whole读出mask，不以fg/bg冒充此次输入。

纠错不是挑正结果；无论小正、小负或更差均保留，不扩扫参。八类新A约39.5%OA，不是旧A+D；本轮不宣称完整模型胜利或唯一因果根因。
