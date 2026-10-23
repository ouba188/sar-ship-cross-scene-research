# 冻结参数外推协议

2026-10-24，执行前记录。本目录独立实现外推，不修改源结果，不训练、不选择任何参数，不进行 Git 操作。原图回看由主代理负责。

## 队列与来源

沿用 local_evidence_transport/external_confirmation 已有3075条观测、tokens、table和三套 strategies。既有源身份排除直接沿用，不重审计全库。这是历史使用过的外部队列，不是全新盲测。仅核对实际使用数组与表的行对应，避免索引错误。

源参考固定为 tokens.npz 与 projection_layer1.npz 中64个参考。源和外部 layer1 使用相同源PCA32、单位化和质量为零时置零的处理。复用 query_evidence_pooling/run_pooling.extract/aggregate；外部索引偏移 sourceN，bank保持源索引。

## 冻结计算

- full = external strategies_seed.logits[:,2]，沿用源实现 float32。
- anchor = full + 源 querypool full_metrics.csv 中 joint 的各seed gain * 外部 joint局部分数 / 源 querypool receipt.scale。
- D 按父 reference_dependence 公式，使用同一源参考置乱与共同有效参考；不调用 prepare，避免外部拟合尺度。
- marginal L 按有效类别中心化，除源 scales.csv 的 single；其两项乘积再除源 nonlinear。D及三个置乱D均除源 joint_shared。
- 五臂 joint/nonlinear/shuffle101/211/307，仅应用 frozen_marginal/weights.npz 中每seed对应7项系数，按类别中心化增量后加入anchor。没有6项marginal系数重训，没有额外倍率。

## 执行与输出

先对少量源样本（包括缺VH）重提取和重放，核对保存的k/a/pi/valid、局部分数、D、7项特征、anchor与五臂预测。失败则定位实现，不运行错误外推；不以外部性能决定修正。

通过后一次性外推3075条，保存 localcache.npz、features.npz、predictions_seedX.npz、小receipt、全量与逐类OA/BA/NLL/Brier、相对full/anchor的救回误伤净数和分组结果。指标使用现有8类softmax规则，无温度校准。所有进程正常退出后交回主代理。结果不佳不调参数。
