# 独立配对分析

本目录只读取冻结预测与源配置，不训练、不选择目标模型。完整解释及保留范围见父目录FINDINGS_ZH.md。

- analyze_results.py独立重算OA/BA/NLL/Brier、逐类/逐港差异和MMSI/过境组bootstrap，并按fold/seed/instance核对旧attention基线。目标旧分数回放误差0。
- source_metrics/contrasts/by_class/by_port和test同名表保留全部臂与三个种子。source验证915个折内位置不是915个独立观测；test每seed4744个唯一观测。
- freeze_strata.py在目标结果前仅用7934条源train并集几何固定长宽/比例三分位数；strata_and_plot.py应用这些边界并输出分组结果与两张完整比较图。
- active_scope.py分开实际非零补充头与源端选0回退。dynamic/fixed四个共同启用组合的BA均优于fixed，但部分总体净收益确有回退贡献。
- learning_diagnosis.py只分析36条已完成轨迹，确认训练损失均下降、源验证晚期退化与J/连续风险选择差异，不用终末模型在测试上回选。
- cases表记录结局首ID，固定共同竞争类优先选实际错误类，并分别保留各模型top1 margin；不因两模型竞争类不同而误称正分差预测错误。

按船和按过境的bootstrap是分别聚类的描述区间，不是联合多层随机抽样保证；不把重复seed扩成独立样本。只有图像对照和固定头干预完成后，才引用对应原图机制结论。
