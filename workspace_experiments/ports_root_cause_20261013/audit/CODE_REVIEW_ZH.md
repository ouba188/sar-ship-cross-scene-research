# 实际实现审阅

2026-10-13。审阅对象：settings.py、extract_source.py、run_factorial.py、analyze_factorial.py 及其调用的历史 geometry/核/支持求解实现。未修改主代码、未运行模型拟合。本审阅使用 karpathy-guidelines 控制范围。

## 结论

当前支持的顺序执行路径中，没有发现需修改主算法代码的可达缺陷。四格源/目标选择、R/T 共用先验、DD 换 dual 先验、支持重求与 equal-softmax compact 融合均与实验设计相符。独立旧 geometry 回放已通过：18 个上下文、108 个臂/读出比较，保存预测完全相同、保存概率最大差为 0。

## 已检查的实际风险

- settings.py:35 的源数组为 38091 全长布局，raw/c4 使用全局行号，extra 在返回时按本折 rows 取子集；未把并集位置误当全局行号。permitted 子集与 ID 断言保留，两折使用各自 source_rows。
- run_factorial.py:13 对 original/display/fold 分开拟合目录；prepare 得到的 bundle.source_rows 与原折允许行核对。目标读取会修改旧 evaluate 模块全局路径，但当前 encode 使用显式传入 bundle、输出使用本地 dst，且下一次 bundle_for 重新绑定 source_arrays/fit_dir；未在实际顺序调用中发现混折。
- run_factorial.py:26 保留原 retained/discarded 尺度和浮点转换。R 仅用第一路核，T/DD 用相同双路核；R/T 共享 r50 prior，DD 使用 dual prior。ridge=8、类别均衡支持系数和 compact 概率融合保持不变。删除 mapper 路径不改变这三个臂。
- extract_source.py 复用三港已完成 O/D 缓存的 5268 个允许行，另提取 30827 行；复用行本身不改变 fit_rows。报告不应写成所有 36095 行均重新编码。
- analyze_factorial.py 的配对 BA、风险符号和交互公式与设计一致。prior 是 expert-only；expert-prior 可作加入支持修正的条件效应，c4_late-prior 还包含 base 路和融合，不能仅归为支持适配。

## 源原图回放失败的边界

原 .02 阈值确实失败，保留 REPLAY_FAILED.json 正确。独立只读缓存比对见 SOURCE_PROVENANCE.json：36095 行只有 1 行 raw 相对误差超过 .02，最大 .0283723711；同一最大偏差已存在于 2026-10-09 pooled_zero 与 legacy pooled_features 的比较。

全部 24324 个 shared-cache 来源允许行的本次 raw 与历史 pooled_zero 完全相同；其 C4 与历史 C4 也完全相同。其余 new-feature 来源中有少量差异。由此可排除“这个最大异常首次由本轮 display 实现产生”，但不能证明历史偏差的具体硬件/精度原因。主线程的有限像素与重编码诊断另有独立记录；本审阅没有再次运行编码器。

因此本轮应以 current-original source refit 为四格基线，单列 archived-source 与 current-source 原图拟合的预测漂移，不把旧缓存当位级复现。无需扩大编码器排查或盲目放宽阈值。

## 数值审计范围

verify_predictions.py 不导入主实验或主分析模块，仅读取原清单和冻结预测。replay、full、compare 阶段均已完成并通过。指定范围独立计算 3240 行指标、3240 行差分和 1080 行救回/误伤，概率归一最大误差 4.10e-8，保存预测与概率 argmax 无分歧。

按主线程最新指定范围，独立四格审计只覆盖 fold2 的 Los Angeles/OSS1/OSS2 和 fold3 的 Port Klang/OSS1/OSS2，全部三臂、三个读出、五个支持 draw，original/common_query 两个范围。重算 BA/OA、宏微 Brier/NLL、逐类召回、三项关键差分及救回/误伤，与 metrics/effects/interaction/task_cells/task_effects 对账。Santos、GRDH 单独任务和新增身份/产品/干扰敏感性不冒充已独立重算。

本文件为代码审阅；数值完成状态以 NUMERICAL_AUDIT.json 与 MAIN_COMPARISON.json 为准。另按后续授权完成 OSS2 产品配对元数据及 fold2 T/expert 的 40 行小范围审计，见 PRODUCT_PAIR_AUDIT.json；总报告见 AUDIT_ZH.md。
