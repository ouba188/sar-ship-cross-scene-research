# 池化独立审阅结论

已结束审阅。基于本目录既有公式检查、source_scale、source_derivatives、selection_replay 和 implementation_checks，不追加扫描或更改主试验。

## 实际结论

- 未发现实际 fold 来源、原通道 W 缩放、双极化求和/C4 平均、源验证 lambda 选择或预测重放错误。18 项源选择均重放一致；四例缓存图的独立 NumPy 池化与积分最大误差 1.78e-15。
- raw W 下，向所有类别权重加同一向量 u 不改变 GAP softmax/argmax，却一般改变 class_residual 的空间权重；contrast_residual 消除此共同空间场。此代数性质成立，但不是物理去噪证据。
- 本次 one-hot、带 intercept 的 Ridge 类目标和为 1，使 sum_c W_c 近零。实际 ||Wbar||/||W|| 为 1.07e-7 至 2.03e-7，全队列 class/contrast delta 最大差 3.75e-6，两臂同净 -56。因此当前读出中去公共项没有独立实验空间，不作为 novelty 或有效机制包装。
- 主线 std 净 +19（OA 81.9842、BA 80.5672），对照 source_mean -1、amplitude-only +6；这些是主线报告结果，不在本分支重复运行。pass 配对区间尚不支持显著收益；点估计仍值得旧 split 复验，不要求每折获胜，也不宣称证明了局部物理机制。

## 尺度与方法边界

当前继承冻结 GAP/PCA/Ridge 的 raw W，不是官方 CSRA 复现。CSRA 的类方向归一化约定不能与 raw W 混同；本次读取的 arXiv/ICCV 版本中 Eq1 是特征提取，空间 softmax 在 Eq2，因此此前“Eq1 L2 归一化”这一精确归属不作已核实引文。独立报告不据未核实实现推断归一化细节。

本方案先减空间均值，得到 pool(s-mu)=pool(s)-mu，因此 A+lambda*pool(s-mu) 与 CSRA 的 GAP 加非中心残差不是相同公式。T=1 的数值也不能消除不同 W 范数带来的温度/残差幅度差。它是同冻结 W 下的公平预定池化对照，不是各方法最佳尺度比较。

源训练 A 的 top2 gap 中位数约 1.53--1.77，class 残差对应类间差绝对值中位数约 0.89--2.21；乘 0.1 后超过原 gap 的比例约 2.31%--7.10%。不能说所有源残差都远超 A。fold1 源验证 class 残差在 lambda=0 的导数为 -0.019502，但 0.1 比 0 差，说明存在小于 0.1 的选择分辨率边界；不修改原网格或补扫。该折 std 导数为正，在固定 delta 的凸 NLL 下非负 lambda 最优为 0，不能同样归因于粗网格。

softmax 中心残差的小响应展开首项为方差，但并不只由 std 决定：既有等均值/等方差反例池化值分别为 0.761594 和 0.503343。std 是有效的离散程度对照，不是 attention 的代数同义词。

三种池化均对联合空间重排不变，不是位置几何创新。4x4/5x5 map 与卷积感受野不足以精确区分 body/bg；不将响应命名为纯船体、纯海面或物理噪声。归一化 W 后重新施加 raw 权重平移通常不保持原 GAP 功能，因此共同平移不变性的论述限于实际有效 raw 线性方向。未建立首创结论。

## 可追溯材料

- `formula_checks.json`, `ridge_common_field.csv`, `cached_map_checks.csv`, `cached_case_deltas.csv`
- `implementation_checks.json`, `selection_replay.csv`, `full_delta_equivalence.csv`
- `source_scale.csv`, `source_scale_by_class.csv`, `source_derivatives.csv`, `raw_W_norms.csv`
- 原论文：Zhu and Wu, Residual Attention: A Simple but Effective Method for Multi-Label Recognition, ICCV 2021, arXiv:2108.02456，方法部分 Eq2--8。

本分支未改主文件、未调参、未重提图、未 Git。下一项仅执行旧 EXP 六折固定 layer3 的 std 复验。
