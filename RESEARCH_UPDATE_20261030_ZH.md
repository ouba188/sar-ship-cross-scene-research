# 2026-10-30：M3定向修正与失败定位

完成三臂×三训练顺序共9次M3-only续训、1377步。M1/M2/主干/分类头保持不变；仅比较器学习率提高、增加3072参数正对角比较度量，均未产生新的分类收益。原完整模型和已完成主表未被覆盖。

- 历史开发160条（3过境、7个出现类）：原完整模型OA45.000/BA35.734；原M3续训、提高比较器LR、新度量三臂均OA40.833/BA34.250，三个顺序逐项top1一致。
- 新度量的开发集同类参考概率VV从0.14937到0.17683、VH从0.14907到0.16525，但不能转述为完整分类改善。
- 只移植新比较器保持OA/BA；只移植新值路径复现16救回、36误伤的事件组成。这定位的是本轮续训退化，不是所有原M3不足的唯一原因。
- 参考等权保持OA/BA；使用查询标签只保留同类参考反而下降。后者仅为非部署机制诊断，不当作主表、不称理论上界。
- 梯度、有限差分、初始化恒等、缺失极化、实际冻结范围、原生chip/缓存与保存重放检查通过。12张新原像素VV/VH面板及真实参考已查看，包含救回、误伤、持续对和持续错。

结论：本轮不升级M3，不扩大3075或公开集，不继续只调selector学习率。保留关系学习正信号与完整分类负结果；下一处应关注参考特异信息如何形成有效回写，以及与分类器的训练协调，而非仅提高相似度分数。

产物：[报告](workspace_experiments/scene_aware_classification_20261018/model_closeout_20261023/m3_relation_repair_20261029/FINDINGS_ZH.md)、[协议](workspace_experiments/scene_aware_classification_20261018/model_closeout_20261023/m3_relation_repair_20261029/PROTOCOL_ZH.md)、[原图案例](workspace_experiments/scene_aware_classification_20261018/model_closeout_20261023/m3_relation_repair_20261029/analysis/cases/index.html)、[独立算子审查](workspace_experiments/scene_aware_classification_20261018/model_closeout_20261023/m3_relation_repair_20261029/audit/AUDIT_ZH.md)。

代码、诊断结果、训练收据和派生图表同步；原数据、完整私人清单、模型权重和预测缓存不上传。实验基于既有开发队列，不是新盲测。
