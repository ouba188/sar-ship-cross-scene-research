# 完整联合训练实现审查

日期：2026-11-17。范围为本目录 model.py/run.py 的实时 source/reference 归一化、计算图生命周期、三组训练配对及早停。仅静态阅读代码、读取已有 smoke.json，以及 CPU 读取既有分组和几何缓存；没有模型前向、GPU、训练、代码修改或 Git 操作。本文件是本次唯一写入。原图审阅留待后续图册，不将本报告当作图像机制分析。

## 结论

在已审范围内，未发现需要阻断本轮训练的实质缺陷。source 主归一化路径使用当前参数实时重算且可微；reference 图每个 microbatch 独立构建，未发现跨 microbatch 复用已释放图的问题。三组共享初始化、样本顺序及训练规则，早停实现符合 max30/min16/patience8。

LocalExchange 的局部 detail 分母沿用旧算子的 stop-gradient。本轮实现保持该定义以隔离训练条件，因此不将其列为待修错误，也不宣称整条归一化路径处处全可微。这不是 source 全面冻结，更不是已经证明的收益不足根因。

## 实时 source 与局部停止梯度

新入口：`E:/临时会话/paper_draft_20261108/pcea_joint_full_20261117/model.py`。

- `Model.__init__` 第18行起直接创建 staged.Network，再调用 cost.ChipPath.__init__；phase=3、fixed_delta=None。没有恢复历史 staged/Full/PCEA 任务 checkpoint。骨干及实际使用的原模块解冻，新 reader/exchange 另行初始化。
- `Model.reference` 第56行起先计算原M3的当前描述子，再用当前前缀、M1、live layer2末端和 CCREL 生成 source raw。第78行直接赋值 `self.exchange.source_raw = self.ccrel.raw`，没有 detach、no_grad 或通过 buffer 复制替代计算图。新 calibrate 第109行为空操作，不读取旧719冻结 source bank。
- `PCEA.standardized` 位于 `E:/临时会话/paper_draft_20261108/pcea_coordinate_repair_20261117/model.py:59`：source raw 经过当前 reader 投影，mean/variance/scale 在计算图内计算，返回的标准化 evidence 使用这些可微张量。
- 同函数第68行起在 no_grad 内将统计量复制到 mean/scale buffer。`E:/临时会话/paper_draft_20261108/pcea_local_cross_20261117/model.py:77` 的 LocalExchange.forward 使用该 `self.scale` 作为 detail 分母。因此 detail 对该分母的梯度停止，但对 local_value、center 的路径仍在，主标准化 token 路径也仍在。
- 同文件第115行 writer 幅度尺度的 detach 也是保留的原算子定义；trace 中 detach 用于诊断。不能把这些局部停止梯度写成整个 PCEA、source 前缀或 reader 被冻结。

本轮明确保留上述算子，不为本次审查改变 detail 分母或 writer 幅度定义。任何“全部归一化路径完全可微”的描述应收窄为“主 source 归一化路径实时可微，局部 detail 分母及 writer 幅度尺度保留 stop-gradient”。

## source32 的实际组成

只读核对来源：

- `E:/临时会话/m1_coverage_20261101/groups.json`
- `E:/临时会话/m1_coverage_20261101/manifest.json`
- `E:/临时会话/paper_draft_20261108/recent_baselines_20261111/native_cache/available.npy`
- `E:/临时会话/paper_draft_20261108/recent_baselines_20261111/native_cache/footprint.npy`
- `E:/临时会话/paper_draft_20261108/full_finetune_20261111/geometry_cache/small/local_slots.npy`

32条参考全部唯一，全部属于719训练样本，与validation索引交集为0；原始类别0/1/2/3各8条。CPU按代码对应的 origin0 footprint occupancy、区域合并及双极化有效条件核对，body/end/side 的有效观察数均为27。

| 类别索引 | 原始参考数 | body有效数 | end有效数 | side有效数 |
|---|---:|---:|---:|---:|
| 0 | 8 | 7 | 7 | 7 |
| 1 | 8 | 6 | 6 | 6 |
| 2 | 8 | 6 | 6 | 6 |
| 3 | 8 | 8 | 8 | 8 |
| 合计 | 32 | 27 | 27 | 27 |

代码 source_weight 使用有效观察等权，而不是训练CE的类别/船只平衡权重；有效子集也不再每类等量。这符合 run.py 当前写明的 equal observation weights、active regions only，不能描述成32条全部参与或类别平衡归一化。

这里是固定源训练池的全局校准，不是逐query执行合法性筛选后的统计池。原M3的query-reference合法性掩码独立保留。使用源池做归一化不等于使用validation校准，也不能将这27条计数本身解释为性能变化原因。

## reference 图生命周期

`E:/临时会话/paper_draft_20261108/full_finetune_20261111/train.py:172` 的 loss_for 每次重新调用 net.reference，再运行当前 query。新 `run.py:157` 起每个batch分成microbatch，每个micro立即执行按样本数加权的 backward，最后统一 optimizer.step。

因此相邻micro拥有独立的source/reference计算图；同一micro内M3与source归一化共享前缀图是正常分支共享，不需要 retain_graph。query再次调用CCREL时，虽然重赋值 ccrel.raw 属性，但 exchange.source_raw 已持有原source张量引用，不会被这次属性重赋值改成query raw。下一micro会重建并替换source引用。

训练时 reference 的M1 hook只在query context非空时记录；reference分支直接计算区域和浅层，不调用query的m2/writer hooks。finish 用 finally 清空context，当前调用顺序未见source覆盖query上下文问题。

验证使用继承的 predict，在固定权重、no_grad 下重算一次reference/source，再复用于该次validation各batch，符合推理场景；未使用validation统计替代source统计。没有发现图被错误复用的静态证据，本次没有自行运行跨micro反向测试。

## 配对、早停和已有 smoke

`E:/临时会话/paper_draft_20261108/pcea_joint_full_20261117/run.py`：

- 第148行起各组采用相同seed派生的独立NumPy顺序与增强随机数；同epoch共用样本顺序、batch增强和训练损失。每micro乘 len(ids)/len(batch)，末尾不满batch的梯度权重正确。
- 第181行早停要求 epoch>=16 且距最佳epoch至少8轮，上限30。最佳按BA/OA/-NLL的词典序选择，epoch0可选；BA/OA不变但NLL改善也会更新最佳并延后停止，符合当前选择规则。
- 三组有相同预算上限、LR候选机会和早停规则，不保证实际训练epochs/steps相同。结果应报告实际消耗，不能写为严格相同步数。各组可以选不同LR，另需保留同LR配对结果以区分选参结果与固定配置差异。

已读取 `E:/临时会话/paper_draft_20261108/pcea_joint_full_20261117/smoke.json`，passed=true。其已有记录显示：三组所检查12条validation的初始logits最大误差均为0；原骨干和原模块通过非零梯度检查；新增reader首步为0、一次输出更新后的第二次反向为非零，符合零输出初始化；reference最大变化约0.46908，PCEA source raw最大变化约0.35356。

这些结果支持初始化对齐、参数更新和动态数据路径正常，但不是全228初始化逐条重放，也不能仅凭reader参数有梯度就宣称每条source统计子路径都已独立验证。尤其不能用 smoke 覆盖上文明确保留的 detail 分母 stop-gradient。

## 范围边界

本轮将719冻结source统计改为32参考池中的27条双极化有效观察实时校准，是明确的新实验条件；mean/local相同。未来本轮内部配对可检验新增分支的相对收益，但与旧实验比较时不能将全部变化只归因于解冻范围。

目前没有训练完成结果或本轮原图救回/误伤分析。局部停止梯度、有效source类别组成和早停差异均为实现/数据事实，不是已验证的失败根因。后续按实际结果与图册审阅，不在此扩大训练、修改算子或提出新的全库审计。
