# 有界验证记录

2026-10-13。

## 上游读取边界

独立代码核对：case_relation_student.py的fit_prior、budget_fusion_source_base.py的prior_oof从完整源池排除HELD船和当前fold全部船；case_completion_class_control.py的fold_model仅用其他fold的multi行拟合。因此新episode只要同属原fold且支持/query船互斥即可复用。未将历史编码器暴露升级成全链路独立确认。

## 实际检查

- expanded覆盖全部1296允许FIT行，且不包含三个目标港。
- 24个local/matched_mixed配对在原fold内，query一致、16支持类别计数一致、各为16不同船，与query船互斥。未匹配失败后按结果重抽。
- 新实现不调用旧fit_probe的w/w.mean()，而直接采用w*(原事件总量/w.sum())，防止固定mass被旧归一抵消。
- 9个旧源单元raw/initial/x共27项重放最大1.43e-15。
- 6个旧选择器固定mass重拟合概率最大误差1.10e-11。
- 新252任务中9个每类均不足4例，采用独立船KFold四分；其他243个走原SGKF。paired_oof内仍验证每重复覆盖一次、支持/验证船互斥。首次失败及修正时点保留在BUILD_FIX_ZH.md。
- 全部44个评估单元的新选择先冻结，之后评价。每种选择器BA增量与避免误伤减丢失救回一致。
- 独立审阅检查原raw、initial、旧两个selector共176行与历史结果一致，最大误差0；按先任务内抽样、后等任务汇总重算当时60行summary，最大误差2.22e-16。
- 审阅指出confidence基线未进总表（已在单独strong_comparators中），随后补入canonical metrics/summary；44个confidence单元与旧结果逐项一致。此变更不重训、不改变任何新选择器。

## 结论边界

BA增益伴随部分明显OA损失；sourceHELD复用同批船；local追加不普遍优于expanded；OSS2-new简单置信度仍更强。支持抽样不是独立训练种子，目标数据已有开发暴露。审阅未发现影响已报告结果的计算错误，不等于已证明唯一根因或完成全数据部署验证。
