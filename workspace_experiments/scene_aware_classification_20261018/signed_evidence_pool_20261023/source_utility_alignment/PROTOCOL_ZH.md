# 固定 signed-only 的源效用 2x2 对照

2026-10-23，运行前冻结。依据：source_utility_diagnosis 已重放 66 头；源验证 balanced NLL 改善但 OA 下降普遍存在，不能只归于测试迁移。这里只改变训练/选择目标，不称新模块。

1. 唯一架构为父 `Correction('signed_only')`，共享 5->16->1、GELU、113 参数，零末层初始化。相同当前 CACHE、原 native 输入形成的 sigma/kappa、source OOF base/inner 矩、outer validation/test 矩、源 OOF 标准化。冻结骨干/读出/划分，不重新抽图或拟合基础模型。
2. 训练因子：balancedCE（父原式，逐类 CE 均值后平均）；jointCE=0.5 ordinaryCE+0.5 balancedCE。无额外 MMSI 权重。六折、seed17/29/43、200 步、AdamW(lr=.003, weight_decay=.01)，全 batch CPU，与父相同 fp32 前向。
3. 每条训练轨迹同时维护两个选点：原 balancedNLL（严格沿用改善超过 1e-8）；J=0.5 OA+0.5 BA，OA/BA 使用比例，含 epoch0，每 5 步检查。J 相同保留最早，绝不以 NLL 打破平局；以整数计数的有理数比较避免浮点假平局。J 与训练损失两个因子独立，合计 36 条轨迹、72 个 selected-head 评价。
4. 只源验证选择。先完成并保存全部轨迹/选点，再统一对测试集进行一次最终评价，不以测试结果调整预算、网格、结构、阈值或选点。原始 A 分数独立保留。epoch0 被选中时原样报告。
5. balancedCE+balancedNLL 的 18 个模型必须复现 `signed_only_retrain` 的 epoch、state、源验证 loss 和测试预测到浮点精度；不覆盖旧输出。保存逐步曲线（含 epoch0 的 val OA/BA/NLL/balNLL/J）、checkpoint、预测、各 seed pooled/class/pass、成本与 replay。
6. 源验证样本数/类数/pass 数、J 单样本分辨率、所选 epoch 和近最佳候选数明确报告。固定已选头的源验证配对块区间只是条件描述，不能消除用同一 validation 在 41 个检查点取最大值的选择乐观性，不作为新增选择依据。
7. 原图：固定 seed17，每个 2x2 单元按 ID 取救回/误伤/CC/WW 首例，去重后用相同原图跨四单元比较；无某结局则报告缺失。VV/VH 分开，尺寸与 OBB 保留，正文不与图重叠。不命名未知物理机制。
8. 写入仅限本目录；无 Git、无新增搜索/臂。结果允许为零或负，来源队列是开发数据，不称独立新数据确认。
