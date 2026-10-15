# 先验保护规则：独立数学测试审阅

本支线仅审查约定接口的数学实现，不复现 CLAP/SS-Text+，不读取实验数据，不作性能或泛化结论。仅写入本报告与 `test_prior_rules.py`；不修改主实现。

## 验证状态

2026-10-12，在 ClearSAR 环境运行 10 项独立测试，全部通过，退出码 0；unittest 报告测试用时 0.014 秒。未发现约定接口范围内的数学实现偏差。该结论仅适用于下述构造与公式，不代表真实缓存重放、预测冻结或识别性能已由本支线验证。

运行解释器为 `D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe`，使用 `-B` 执行 `E:/临时会话/prior_protection_20261012/test_prior_rules.py`。CUDA 对此进程不可见，数值库线程环境变量设为 2；未使用 GPU，禁用字节码写入，不生成测试缓存或侧文件。

## 独立参照与覆盖

对每个有限正则列，在特征空间直接求解

`delta_c = (X.T W X + lambda_c I)^(-1) X.T W (2 onehot(y)_c - 1 - prior_c)`，

其中 `w_i = B / (num_present * count(y_i))`。该参照使用 D×D 的 primal 正规方程，不复写待测 B×B dual 求解路径；无穷正则列定义为严格零。小维度构造含不均衡标签、非零 prior、D<B / D>B 与线性相关特征列。

测试覆盖五种 penalty 的手算值；各模式对 primal 的一致性；完全覆盖、均衡标签且各类真标签平均置信度相等时退化为旧 ridge；缺失类严格零列与 guard 已出现类保持 raw 解；支持集排列不变性；softmax 平移不变性；confidence 对高置信类别给更强正则，并在固定该列 RHS 的比较中缩小残差范数；单个已出现类时的样本权重；零残差不保证概率不变的反例。

这里的 count 使用全部 C 类计算 B/C，样本权重则使用 num_present；两者不能混用。confidence 的分母是已出现类别平均置信度的等权均值，不是按样本数加权的总体置信度。

代码阅读确认 confidence 对数值非正的已出现类均值显式报错，不静默加 epsilon。本次正常概率与大幅行 logit 平移构造均通过；未使用真实缓存验证该异常是否会出现。

## 必须保留的解释边界

缺失类别 `delta_c = 0` 只保证加性 readout 路径中该类别的原始 logit 不变。其他类别 logit 改变后，softmax 的共同分母也会改变，因此缺失类概率、类别间 margin 与最终决策仍可能变化。测试含可手算的三类反例：原 logits 全零，guard 调整为 `[0.5, -0.5, 0]`，缺失类 logit 仍为 0，但其概率由 `1/3` 变成 `1/(exp(0.5)+exp(-0.5)+1)`。不能把已有“零残差列通过”写成“概率完全保护”。

confidence 的“更强 anchor”指更大的正则惩罚，以及固定 X、W、该列 RHS 时更小的残差范数；不承诺每个查询样本的 logit 改变量单调下降，也不承诺识别性能提高。

penalties 接口只接收 support labels、support prior、固定 base_strength 和 mode。局部数学测试不能独立证明主调用方传入的确实是 support 切片，也不能证明整个实验不存在 query 信息泄漏；该集成边界不应由本测试替代。
