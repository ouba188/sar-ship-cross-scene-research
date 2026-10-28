# 冻结扩围代码与资产

本轮只评价既有模型，没有新训练、选epoch或倍率搜索。原先训练时仅新增M2和既有layer3.5三个卷积更新；现在全部权重冻结用于推理。

## 输入

- `../m2_residual_placement_20261028/checkpoints/preadd_{17,29,43}.pt`：固定epoch6的C候选delta。
- `../shallow_m2_expansion_20261028/predictions.npz`：基线/原位置既有3075分数。
- `../shallow_m2_writeback_20261028/checkpoints/`：六个旧模型，仅用于抽样重放确认复用可比性。
- 既有扩围query_records/query_metadata/coverage以及current发布包teacher权重、16条参考和旧development清单。
- 同清单指向的原始VV/VH图片及OBB信息。没有新增下载或修改标签。

## 执行

```powershell
& 'D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe' -X utf8 -B 'scene_aware_classification_20261018/model_closeout_20261023/m2_preadd_expansion_20261028/run.py'
& 'D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe' -X utf8 -B 'scene_aware_classification_20261018/model_closeout_20261023/m2_preadd_expansion_20261028/analyze.py'
& 'D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe' -X utf8 -B 'scene_aware_classification_20261018/model_closeout_20261023/m2_preadd_expansion_20261028/case_diagnosis.py'
```

`run.py`从原生图片重新生成输入，计算冻结的layer2前三个block输出作为C的新输入，原始layer2/anchor3继续作为M3比较锚点。查询/参考值使用同一模型，不能把新查询与旧参考值混用。

开始时在3075队列中均匀取16条重放6个旧对照，另用旧160开发队列的16条重放3个C模型。确认数值一致后复用旧完整结果，只计算C三个模型的新分数。日志明确3个新推理和6个复用，不称9次新训练。16张分块保存以便中断恢复；最终按有序ID与标签合并，不按文件排列猜测对应关系。

`analyze.py`CPU分析，按船只/过境分组、固定模型条件性bootstrap和实际参考生成原图面板；`case_diagnosis.py`读取其结局选例，固定C权重，关闭骨干传播/直接读出/极化/区域路径。关闭同时作用于查询和参考，没有重训，不能解释为物理因果。

旧实验不覆盖。公开归档沿用current包路径，而原机使用同内容standalone可选模块；仅归档runner的包搜索路径按既有规则改为current。模型权重、原数据、特征缓存、私有完整清单和逐条推理chunks均不上传。
