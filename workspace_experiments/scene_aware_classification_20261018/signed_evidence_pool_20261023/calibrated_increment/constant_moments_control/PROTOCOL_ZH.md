# 固定校准 B 之外的常量矩对照

2026-10-23，运行前固定。只跑 signed-only 常量矩：new/old 两划分 x 6 折 x seed17/29/43 = 36 条轨迹。目的为区分逐图 sigma/kappa 与“逐类固定参照 + B 分数门控”，不是新增模型或按测试寻优。

复用父 `run_increment.load_fold` 与 `ref`，不修改父文件。每个 fold 的固定矩为 `oof[train].mean(axis=0)`，形状 3x2；沿样本轴平均，保留三个类别读出方向，不按样本真标签分组。训练、源验证和测试都广播同一矩：descriptor 中 sigma、中心化 sigma、kappa、中心化 kappa，以及最终乘法载体 kappa 均常量化。不是只常量化门控输入而保留真实乘法信息。

冻结父同 partition/fold/seed 的既有 score-only 校准 C，B=C(A) 不变。新增头使用 B 中心化分数、固定矩和其源训练标准化；标准化规则与父一致，统计仅由本轮 source OOF train 输入计算。仍为共享5->16->1/GELU/tanh*kappa、113 参数、零末层，但有效逐图输入仅剩 B；固定类别矩可以提供类别参照，不能称输入信息/有效容量与逐图版本完全相同。

原源 OOF base/inner 对应关系、fold、C、训练标签、jointCE=.5ordinaryCE+.5balancedCE、AdamW .003/.01、200步、seed、每5步检查均照父。源验证主J=.5OA+.5BA，含epoch0，有理数比较、平局最早；不追加次要选点。所有36选点保存后统一评价test，不用test改参。

边界继承：基础读出OOF但C非OOF，残差源监督不称样本外错误监督；validation先选C再选新增头，存在选模复用；两划分为开发队列，不是独立确认。报告均为B之外的增量，不把A->B贡献算作矩信息。

检查固定矩来源、descriptor/载体沿样本零变化、0步=B、对应C/B重放、存盘与预测重构。保存曲线、模型、常量3x2、逐seed/fold/class/pass与相对父逐图signed、extra-score的配对（pass/MMSI条件区间都报）。固定seed17每划分四结果原窗回看，可复用renderer；必须传baseline_label='B (frozen calibration)'与明确subtitle。只写本目录，无Git、不改父run_increment或其他代理扩展。
