# 本轮复现与资产边界

本轮为M2位置的独立研发实验，不覆盖 `sar_m123_current_20261027` 默认模型。新增算子位于 `sar_m123_standalone/sar_m123/block_m2_placement.py`，复用既有ShallowM2的完整M1/M2/M3后端和训练损失。

## 输入与依赖

- 当前发布包中的teacher权重、源适配136条、开发160条、参考16条以及历史干预选样清单。
- 相邻目录 `shallow_m2_writeback_20261028` 的原生图像前缀缓存、6个已训练baseline/layer2对照、run.py匹配训练函数。
- 真实chips仍由原输入清单指定。源/开发/参考、图像裁边补零、VV/VH可用性和区域掩膜不变。
- 本轮新缓存只增加到layer2最后一个block之前的固定特征，不包含学习后的block输出；M3比较保留旧锚点，内容路径实时计算。

## 执行

在本目录使用项目ClearSAR环境运行：

```powershell
python -X utf8 -B run.py --mode prepare
python -X utf8 -B run.py --mode train
python -X utf8 -B analyze.py
python -X utf8 -B diagnose.py
```

prepare确认当前代码、缓存和原生block的行为。train训练6个新的pre/post位置版本，复用6个已核对的baseline/stage结果。stage是上一轮layer2版本的报告别名，不是另一次新训练。固定末轮结果，不做基于开发集的epoch或参数选择。三个seed改变训练顺序，M2初始化保持位置间匹配，不是三套独立预训练初始化。

检查点使用当前实验run.py的construct和receipt中的position恢复。增量同时含新增M2与已解冻的layer3.5三卷积；仅加载新增M2不足以重现本轮结果。run.py复用旧训练函数，因此旧checkpoint字段stages在内部位置时存的是位置字符串，实际恢复依赖本轮construct，不应交给旧ShallowM2的stage列表接口。

## 诊断含义

diagnose.py保持同一模型权重，分别关闭骨干传播、末端直接读出或VV/VH/区域回写。区域/极化关闭同时作用于查询和参考；它检验算法路径依赖，不是图像像素干预，也不能单独命名物理原因。输出path_predictions.npz留本地，path_diagnosis.json供审阅。

原图审阅按实际VV/VH及参考组合进行。代表案例不是随机样本，不用案例数量估计总体比例。指标表和原始正负结果均保留。

Git归档只含代码、统计、报告和少量派生案例图。原始数据、权重、缓存、私人输入清单以及下载的论文PDF/全文不上传。归档代码的包路径改为已有 `sar_m123_current_20261027`，其中归档可选模块与本地新增模块相同；研究脚本仍需上述本地资产，不宣称一条命令可下载全部数据。
