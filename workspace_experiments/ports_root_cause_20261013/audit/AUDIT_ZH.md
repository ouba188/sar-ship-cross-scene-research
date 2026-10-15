# 独立实现与数值审计

日期：2026-10-13。状态：完成，主 CSV 对账通过。NUMERICAL_AUDIT.json 的 main_csv_comparison 已为 passed，详见 MAIN_COMPARISON.json。未发现审阅范围内需要修复的主实现缺陷。没有运行模型拟合、编码器重提取或修改主文件；所有写入均在 audit/。

## 覆盖与数值一致性

1. 旧 geometry 回放覆盖全部 18 个上下文，R/T/DD 的 expert/c4_late 共 108 个比较。保存预测完全相同，保存概率最大差为 0；R/T 的 expert-only prior 完全相同。
2. 新四格按主线程指定，覆盖 fold2 的 Los Angeles/OSS1/OSS2 与 fold3 的 Port Klang/OSS1/OSS2。24 个四格上下文，三臂、三个读出、五次支持抽样，original/common_query 两个范围。另计算 12 个对应 archived 对照。
3. 独立脚本从清单按 sample_id 取标签，直接读冻结概率和预测，不导入主分析的 metrics、分组或配对函数。验证查询/支持 ID 相同，指标按真实出现类别计算。

| 对账对象 | 行数 | 最大差 |
|---|---:|---:|
| 主指标 | 2160 | BA 0；OA 1.43e-14 点；宏 Brier 4.45e-16 |
| 逐类记录 | 17280 | 召回 0；Brier 6.67e-16 |
| 差分与交互 | 3240 | BA 2.14e-14 点；宏 Brier gain 7.01e-16 |
| task_cells / task_effects | 432 / 648 | BA 1.43e-14 / 8.89e-15 点 |
| 救回减误伤 | 1080 | 净 BA 1.43e-14 点 |
| original 源拟合漂移 | 540 | BA 0；OA 1.43e-14 点 |
| 新增关键类别贡献 | 120 | BA 贡献 8.89e-16 点 |

保存概率归一最大误差 4.10e-8；保存预测与保存概率 argmax 无分歧。独立风险差 delta_brier 定义为新减旧，主表 risk_gain 定义为旧减新，比较时已显式反号。

新增 class_effects 只复核原六任务 R/expert 的渔船、原油轮两类及协调/净收益两个对比，复用已审计逐类结果，未重新推断。Santos、OSS2_GRDH 独立任务及其他敏感性没有全部独立重算。独立表 ports3 连接键只含 LA/Klang，不能当完整三港汇总；汇总对账使用 task_cells/task_effects。

## 对根因解释的核对

当前证据支持“源端拟合不匹配是冻结管线的一项可干预原因”，不支持唯一根因或已全面修复跨域失效。重要解释边界如下：

- 主表三港 R/expert 协调恢复 +5.446 点，匹配净收益仅 +0.214 点；渔船恢复贡献 +5.003 点。恢复主要针对处理诱发的类别损失，不能把 +5.446 称为相对原始协议的净提升。此三港汇总含未独立重算的 Santos。
- OSS1 R/expert 匹配净收益 +4.324 点，原油轮贡献约 +4.416 点，且该类全队列仅 13 行。净收益可成立，但主要由稀少类别驱动，重复 draw 不增加独立船舶或训练种子数。
- OSS2 R 源协调在 prior 上 +1.43255 点，在适配 expert 上 -0.06996 点：源先验改善没有转成同量的支持适配收益。不能据 prior 结果宣布最终系统修复。
- prior 仅为 expert-only。expert-prior 是加入支持修正的条件效应；c4_late-prior 还引入 base 与融合，不是纯支持适配贡献。
- 逐裁片 display 同时改变灰度尺度、偏置、截断和取整；目标支持与查询同时处理、支持解重求。源模型各组件也同步重拟合，未单独识别物理显示因素或某个源组件。

## 源缓存异常

保留旧 .02 阈值失败正确：36095 行仅 1 行 raw 超阈值，最大 .0283723711；同一最大偏差已存在于 2026-10-09 pooled_zero 对 legacy pooled_features。24324 个 shared-cache 来源允许行的 current raw 与历史 pooled_zero 完全一致，具体历史数值原因仍未定位。当前对 archived 原图拟合的 540 个 BA 差范围为 -0.71925 至 +1.32614 点，不能笼统视为零；正式四格以 current-original 为基线、漂移单列，做法正确。

## GRDH/SLC 配对核验

独立重解析产品 datatake 和原始 SAR 时间字符串，重建 ID 对集合：370 对唯一 identity+datatake，234 对时间一致，233 对标签一致，1 对冲突；身份和类别与原清单一致，未重解析 XML。实际同标签双查询对为 227/223/224/222/223，both_clean 为 156/151/154/153/152。

OSS2 fold2 T/expert 四格、五 draw、all/clean 共 40 行对账通过；BA 差最大误差 3.56e-15 点。它是同支持的自然产品配对，仍混合处理、显示和几何差异，不能命名为随机化产品因果效应。OSS2 精确 spacing 元数据缺失，未做新的 SLC 几何拟合，也不能用像素数推断已匹配物理尺度。两个公开释放相关、GRDH 为 OSS2 子集；上游预训练暴露及支持/查询共享身份边界仍在。

## 文件

- `verify_predictions.py`：replay/full/compare 三阶段，已完成；`REPLAY_AUDIT.json`、`NUMERICAL_AUDIT.json`、`MAIN_COMPARISON.json` 保存范围与误差。
- `verify_product_pairs.py`、`PRODUCT_PAIR_AUDIT.json`、`independent_product_pair_metrics.csv`：配对核验。
- `verify_class_contributions.py`、`CLASS_EFFECT_AUDIT.json`：新增 120 行关键类别贡献核验。
- `probe_source_provenance.py`、`SOURCE_PROVENANCE.json`：只读缓存来源比对；未拟合或重新编码。
- `CODE_REVIEW_ZH.md`：实际接入代码审阅；`DESIGN_REVIEW_ZH.md`：较早设计建议，当前审计范围和读出定义以本报告为准。

以上验证完成后没有扩大或重复主试验。全部本审计启动的进程已正常退出。
