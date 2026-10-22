# 中心锚定必要性：固定两臂对照

2026-10-23，运行前固定。仅本目录写入，不改父目录代码或冻结输入，不 Git。

只比较 uncentered_external 对父 anchored，以及 uncentered_own_full 对父 own_regions_full。保留父 Anchored 网络 79->32->32、32 原通道组、.5*tanh、末层零初始化、200 步、种子 17/29/43、AdamW(.003,.01)、类别/MMSI 权重、每五步源验证 balanced NLL 选步、1e-6 改善阈值及 0 步回退。外部激活范围沿用 has，own-full 激活全部目标，不重选范围。

唯一训练设计变化：贡献 q_g=W_g(h_g-h0_g) 改为 qraw_g=W_g h_g=q_g+origin_g，origin_g=W_g h0_g。原 baseline 分数 base/oof 原样读取父缓存，不用重拟合分数替换它们。

每个 outer 和每个源 OOF fit 都调用 run_strong.View/combine/fit_head，保持 float32 特征拟合及同 selected kind/alpha。显式 W 用 float64，由 standardization、PCA、post scale、ridge 及组合视图归一化链构造。h0 是对应 fit 原特征均值，用 float64 累积。各 OOF held 只分配其 fit 的 origin，不借用 outer origin。

原分组不变：stem VV/VH 各8组，每组8原通道，mean/max 对应通道绑在同一组，共16；deep 原通道按固定顺序分16组。未选 stem 的折前16组贡献及 origin 为零。不按 PCA 轴分组。

每个 fit 核对 centered group sum + A(h0) 以及 raw group sum + [A(h0)-sum(origin)] 均与直接 A 线性读出一致；核对重拟合 centered q 与父 q/oq，baseline top1 不变。保留父 env 和 own env 缓存，OOF 仅 baseline/group reader 按父逻辑重拟合，不声称 env 的整条投影链也重新做了内层 OOF。

每折保存 origin/W/h0/fit/held 和新 qraw/oqraw，保存36头、测试逐例预测和 pooled/配对表。逐例对比相同 seed 和 ID，三种 seed 是同队列重复优化，不算三份独立数据。至少查看救回/误伤/CC/WW 原始双极化图；父图可覆盖时只引用已有图，注意父图文字是父模型结果，新结果以本支线表为准。

代数上同一门控系数时，uncentered 增量比 centered 多 sum_g a_g origin_g；这是与门控耦合的类别偏置项，不是独立 class-bias 分支。两臂重新训练后的门控系数也可能不同，配对性能不能被解释为固定系数下的纯偏置干预。该实验允许特征门控隐式改变有效原偏置，不声称与所有 SE 网络完全等价。不预设中心锚定应优。
