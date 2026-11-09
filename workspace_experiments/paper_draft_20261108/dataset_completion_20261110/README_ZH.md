# 当前组件的必要数据集补充

本轮沿用已完成OSS2分阶段实验的实现、日程与学习率倍率，只补OSS1和自建24港来源的当前算子组合。历史实验不重跑，正式模型与论文正文不自动替换。

## 入口

使用现有ClearSAR环境，在本目录执行：

```powershell
python extend.py prepare oss1
python extend.py smoke oss1
python extend.py train oss1
python extend.py evaluate oss1
python results.py analyze oss1
```

24港将末尾数据集名改为 `ports24`。已完成的缓存、训练阶段和评价预测会直接复用。`smoke`只检查形状、梯度与四类原实现一致性，不是精度实验。不要为更新报告重复运行训练；已有预测只需执行 `results.py analyze`。

本地环境路径：`D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe`。本目录脚本沿用相邻 `experiment_completion_20261109`、`staged_ablation_20261109` 和现有模型工程；它是复现实验入口，不是第二份独立模型发布包。

## 结果定位

- `PROTOCOL_ZH.md`：运行前固定的范围、对照、划分与统计口径。
- `oss1/`、`ports24/`：系统结果、逐类/场景统计、训练阶段记录及本地原图案例。
- `metrics.csv`：每个配置、每个种子的完整评价。
- `paired_intervals.csv`：按类别分层、按船聚类的配对区间，条件于已有场景和已训练模型。
- `per_scene_differences.csv`：按平台绝对轨道统计，不将产品切片当成独立过境。
- `confusion_counts.csv`：逐种子的真实类别、预测类别与数量，可直接复用绘制混淆矩阵。
- `paired_changes.csv`：完整模型与对照的救回/误伤数量；其中M3开关是同权重诊断，不是重训删除消融。
- `source_development_classes.csv`、`stage_metrics.csv`：源端拟合与开发表现，辅助检查覆盖、拟合和阶段变化。
- `training_budget.csv`：各阶段步骤、样本暴露、训练参数量与计时。计时不包含此前缓存准备及全部初始化，不是端到端部署时延。
- `index.html`、`case_records.json`、`cases/`：本地原尺度VV/VH、实际合法源参考及预测分差；示例参考不冒称贡献最大者。

大型输入缓存、权重与逐样本预测在 `C:/Users/25791/AppData/Local/sar_paper_experiments_20261110/`，不提交Git。源数据、逐样本元数据及原图面板亦不随本轮代码/聚合结果归档。

## 不能混用的结果

1. 既有OSS2六臂配套重训与当前工作版OSS2到OSS1外推已完成，直接沿用。
2. 本轮是三个配置、三个种子的各数据集源端训练，采用共同ImageNet起点。不是把旧版检查点改名，也不是逐步重演历史结构变更。
3. OSS1评价为既有四类63艘船、单过境；不是全OSS1多场景结论。
4. 自建24港来源队列评价为八类3075条、1950艘船、18个港口；不是24折留一港。开发160条中97条与源端绝对轨道不重叠，另行报告；评价3075条均无源/开发同船或绝对轨道重叠。
5. F_0921历史结果保留。本轮未将其七类旧支持适配任务改成当前源端训练任务，不能写成已完成当前版本复现。
6. 所有组采用各自教师的干预适配，不能从本轮单独推断干预选样净增益。SE是两个匹配位置的标准注意力对照，不冒称其他论文的完整网络复现。

完成状态以各数据集结果报告和本地receipt为准。
