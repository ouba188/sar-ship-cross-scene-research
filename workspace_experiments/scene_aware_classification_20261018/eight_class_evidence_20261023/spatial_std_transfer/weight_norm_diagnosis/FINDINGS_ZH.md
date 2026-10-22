# 八类 std 权重范数诊断

已完成，只读父 W/std 缓存。结论：**std 确实混合分类器范数与单位方向响应，但本轮 crude/offshore 的主要额外类别偏好不能笼统归因于 W 范数较大。** 尤其相对受损 chemical 类，范数项反而抵消一部分增量优势。不能因此自动建议归一化 W，更未评价归一化后的新模型。

## 精确关系

实际算子先减类公共场，故精确有效向量为 `w_cp=W_cp-mean_class(W_p)`。每个极化、每个 rotation，有

`weighted_std(w_cp · F) = ||w_cp||_2 * weighted_std((w_cp/||w_cp||_2) · F)`。

范数与 rotation 无关，所以同样可提出 C4 平均之外；先分极化分解，再按真实有效 flags 求和。不能将 C4 平均 std 偷换成跨 rotation 联合 std。缺失 VH 保持 0，不影响 VV；valid-only 均值与计入缺失零值的积分均值分别保存。

本次 raw W 与有效 w 的范数最大相对差仅 2.41e-7，Wmean/Wrms 最大 2.14e-7，因此口头用 raw W 近似不改变数值结论，但“精确”等式使用有效 w。从缓存 r map / ||w|| 重算 weighted std，再乘范数并 C4 平均，重放 delta_pol 最大误差 2.22e-16。未重提 F。

这里的单位方向响应仍含特征协方差、原图内容、有效 footprint 和卷积 RF 的影响，不是“真实类别证据”标签。写成协方差形式是 sqrt(w^T Sigma w)，既含方向也含长度；原通道范数还依赖特征坐标尺度，不能直接解释为分类置信度。非负 std 加到分数，也不保证其相对增量有利于真类。

## 源训练结果

仅用每折 source train 做诊断，全部源样本与每个真实类别分别统计。下表跨折按样本暴露数加权；13229 是 fold-sample 暴露数，源样本可跨折重复，不是独立样本数。

为描述两因素，使用对称精确记账：对两类 c/b，`n_c*u_c-n_b*u_b = (n_c-n_b)*(u_c+u_b)/2 + (u_c-u_b)*(n_c+n_b)/2`。分别叫范数项、方向响应项，并对两极化求和。它不是唯一因果归因，也不是运行新 arm。

| 源全体方向比较 | raw delta 均值差 | 范数项 | 单位方向响应项 |
|---|---:|---:|---:|
| crude - bulk | +0.235430 | -0.054353 | +0.289783 |
| crude - chemical | +0.448504 | -0.083012 | +0.531516 |
| offshore - bulk | +0.128209 | +0.046267 | +0.081942 |
| offshore - chemical | +0.341284 | -0.022187 | +0.363471 |

前三项中仅 offshore-bulk 的范数项正向贡献明显，约占净差 36.1%；不能把这个比例推广到全部类别。存在抵消时 signed 比例可负或超过 1，不能当“解释方差百分比”。所有分折/分极化/分真实类结果保留在 CSV。

真实 chemical 源样本内，crude-chemical 差 +0.370331 = -0.085101 + 0.455432；offshore-chemical 差 +0.310330 = -0.036104 + 0.346434。即在主要受损类别自身，方向响应差已存在，范数不是其正向偏好的主因。真实 offshore 源样本内 offshore-bulk 为 +0.282794，其中范数 +0.125537、方向 +0.157257，仍须保留分群差异。

以 fold0 为例，VV 的 chemical/crude/offshore 范数分别 13.753/12.173/6.916，但有效源均值 raw std 为 0.557/0.840/0.699，单位方向 std 为 0.0405/0.0690/0.1011。VH 的对应范数为 21.972/19.314/18.230，raw std 为 0.667/0.793/0.801，单位方向 std 为 0.0304/0.0410/0.0439。更大的范数不等于更大的实际响应。

三折 VV 的范数与八类源 raw 均值 Pearson 为 [-0.320,0.135,-0.151]，VH 为 [0.633,0.232,0.534]；这只是每折每极化八个类别方向的描述相关，既不是预测模型，也不能以相关值单独认定因果。

## 目标集仅描述

4744 条互斥 test ID，未用其标签选择缩放、阈值或实验臂。对应四项 raw gap / 范数项 / 方向项为：

| 比较 | raw gap | 范数项 | 方向项 |
|---|---:|---:|---:|
| crude - bulk | +0.203261 | -0.053212 | +0.256473 |
| crude - chemical | +0.401046 | -0.079575 | +0.480621 |
| offshore - bulk | +0.131078 | +0.050732 | +0.080346 |
| offshore - chemical | +0.328863 | -0.024780 | +0.353643 |

真实 chemical test 内 crude-chemical 为 +0.253555 = -0.067958 + 0.321512；offshore-chemical 为 +0.303865 = -0.018899 + 0.322764。符号与源诊断一致。真实 bulk test 内 offshore-bulk 是小正 +0.014675，由范数 +0.060215 和方向 -0.045540 抵消得到，这是范数确实可能改变方向偏好的具体边界。

这些是未乘原 lambda 的 delta 差，不是 OA 损失的因果分解，不能将上述比例当净 -94 中由范数造成的比例。本轮未生成 normalized score、未重新选择 lambda、未计算新 arm 的预测或指标。对 source utility 的有符号诊断由主线处理。

## 交付与范围

`diagnose_norm.py` 已运行，主体约0.40秒；0 fit、0 新臂、0 选择、0 GPU 提取。输出：weight_norms.csv、source_target_direction_stats.csv、direction_correlations.csv、pair_factorization.csv、pooled_pair_accounting.csv、factorization_checks.csv、cached_map_replay.csv、summary.json。

源来自 baseline/models/fold*_outer.npz 和 baseline/fold*_inputs.npz，统计来自 spatial_std_transfer/std_features.npz 与既有 validation_maps.npz；cohort/valid/splits 均未改。本轮是缓存读出诊断，不是新增原图实验；既有输入/图像边界沿用父报告，不重新挑图。旧 split 报告也已完成。无后台计算、无父目录编辑、无 Git。
