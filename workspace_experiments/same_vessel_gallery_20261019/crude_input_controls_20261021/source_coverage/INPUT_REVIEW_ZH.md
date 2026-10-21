# 五臂输入对照定向审阅

补记（2026-10-21）：用户随后确认 perturb 已完成，baseline 重放误差 `<1e-5` 断言通过。下文“待主流程执行”是审阅当时的状态；本补记依据用户执行回报，不重复运行或扩展审阅。

2026-10-21。只读审阅主目录 `PROTOCOL.md`、`input_controls.py` 及其实际调用的 `masks_for`、`crop_pad`；仅写本报告。未读取特征大数组、运行提特征/拟合/扰动、修改主文件或干扰正在运行的任务。

**结论：在指定五项范围内，未发现会改变实验结论的实现或协议错误，无需因本次审阅停止当前特征提取。** 实际特征提取检查和 perturb 数值重放尚不能由本次只读审阅宣称已通过。

| 检查项 | 结论与依据 |
|---|---|
| preconvmask 隔离 | `input_controls.py:74` 在归一化及 stem 前计算 `x*target` 或 `x*(valid-target)`，且 stem 为 eval、推理阶段禁用梯度。固定 mask 时，body 臂无法读取原始外部像素值。归一化后的黑色常量及硬边界仍影响响应，但不构成原始外部内容泄漏；协议已说明人工掩膜及边界可见。`full_target` 的卷积感受野可以读取外部，协议也明确承认。 |
| pool 维度 | `input_controls.py:61` 的 mean/max 均沿空间维 `(2,3)`，输出各 64 通道。160 输入经过 conv1 stride=2，再 avg_pool2d(2)，得到 40×40；footprint 经 avg_pool2d(4) 同为 40×40。每极化 128，双极化 256，加四个形状量共 260，与缓存和拟合一致。四次旋转同时作用于输入和 footprint，最后均值。 |
| 同划分与源端选择 | `input_controls.py:158–178` 在原 `prior.indices(frame,fold)` 后应用同一个 legal 条件，五臂共享 train/val/test 与 `prior.weights`。各臂仅用本折 train 计算 mean/std、拟合三个 alpha，并以 val BA/Brier 选择；没有用 test 对错选臂或选 alpha。只读元数据检查证实 VV target mask 没有造成额外删样本，既有 source_coverage 的划分仍适用。 |
| 缺 mask/VH | `extract` 对缺 target mask 或 `disable_new_vh` 的 VH 同时保留零响应、零形状，五臂一致；VV 缺 mask 则整条样本统一排除。`perturb` 使用相同条件及同样的零初始化。零值在后续 source 标准化后通常不是零，因此准确口径是“原始 VH 描述子置零”，不是声称模型完全无法识别 VH 缺失；这是共享处理，不破坏臂间匹配。 |
| perturb 重放 | `input_controls.py:232–283` 复用相同 prepare_view、describe 和形状拼接，按 `row.fold` 读取对应 `full_global` 头；标准化、sqrt(260) 缩放、coef/intercept 重放与 fit 相同。仅 target 或 rest 区域加固定位移，rest 使用既有合格 other-mask 的补集，不动 target/other/padding。clipcounts 的两个槽分别累加双极化的 clipped 数和区域像素数，比例计算正确。零位移 baseline 会与既有 full_global margin 比较，要求误差 <1e-5；该断言需主流程实际执行后才算重放通过。 |

## 本次实际核对的元数据

使用已有 `masks_for` 原函数，仅由快照多边形生成 mask，不读原图。所有检查对象的 target/other/rest 构成互斥完整分区，故 `valid=mask.sum(0)` 正确；crop_pad 对填充区的三类 mask 均置零。

- cohort 为 15,252 行；排除 31237 后，无 VV target mask 缺失。VH 缺 target mask 1421 条，关联禁用 47 条，两者无重叠；这些是整个 cohort 的数量，不是测试集数量。
- fold 0/1/2 的 train、val、test 额外 VV 排除数均为 0。测试数保持 1561/1632/1551，共 4744。
- 三折 train 中置零 VH 分别为 487/493/505，val 为 36/21/25，test 为 103/103/102。这些处理在所有臂相同；缺失组的存在不能被报告为纯双极化结果。
- 本轮 100 原油轮对的 200 端点均有可用 VV/VH，不受上述置零处理影响。

池化 footprint 的“排除 padding”是不给纯 padding 空间单元权重，不是保证边界特征感受野不触及 padding；同理 target pooling 不是卷积前隔离。当前协议的主要因果边界已经区分这两者，无需把这一正常实现语义报成错误。

审阅没有扩展到全量数据审计、文献或原图，也没有据代码中存在断言便声称运行结果已验证。待主流程完成后，其既有 input_checks 和 perturb baseline 重放结果应作为执行证据。
