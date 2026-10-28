# 复现本轮等宽位置实验

## 本轮新增与保留

可选算子：`sar_m123_standalone/sar_m123/block_m2_placement.py` 的 `Layer2BlockM2` 新增 `preadd`、`postadd`，原 `none/stage/pre3x3/post3x3` 分支保留。没有更改当前发布模型默认入口或权重。

`run.py`复用上一轮runner的合法缓存与构造函数，以及浅层回写实验的同一训练函数。`diagnose.py`只分析固定训练后的路径，不再优化权重。`analyze.py`仅CPU分析logits及生成原图参考面板。

## 本地资产

- 当前发布包的teacher权重、source_adapt/development/references清单及selection_archived。
- `shallow_m2_writeback_20261028/cache/`中的源/查询/参考缓存。
- `m2_block_placement_20261028/cache/*_prefix2.npy`，固定layer2前3个block输出。
- 前两轮baseline与stage预测、训练收据、delta权重，用于6组对照复用和当前代码重放。
- 原始VV/VH图片，供结果图库逐条追溯。原始数据、完整私有清单、缓存和模型权重不上传GitHub。

## 执行顺序

在工作区根目录，用既有ClearSAR Python执行：

```powershell
& 'D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe' -X utf8 -B 'scene_aware_classification_20261018/model_closeout_20261023/m2_residual_placement_20261028/run.py' --mode prepare
& 'D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe' -X utf8 -B 'scene_aware_classification_20261018/model_closeout_20261023/m2_residual_placement_20261028/run.py' --mode train
& 'D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe' -X utf8 -B 'scene_aware_classification_20261018/model_closeout_20261023/m2_residual_placement_20261028/diagnose.py'
& 'D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe' -X utf8 -B 'scene_aware_classification_20261018/model_closeout_20261023/m2_residual_placement_20261028/analyze.py'
```

训练从同一teacher重新开始，非从前一位置继续训练；已完成且保存的本轮模型可直接复用。清单有序ID必须与缓存一致。保存格式沿用上一runner，`stages`字段此处保存位置字符串，必须由本轮`construct`重建，不可传给旧`ShallowM2(stages)`加载器。

`prepare`检测插错位置、跳过新模块、冻结层误断梯度、复用缓存/模型不一致等具体失败；非零回写与原生BN3后hook/第三次ReLU前hook比较，初始恒等不足以替代此检查。训练末尾验证冻结状态及保存模型重放。

公开归档沿用之前current包路径；算子同内容归入`workspace_experiments/sar_m123_current_20261027/sar_m123/`。原机使用standalone包；前一轮归档runner已处理该路径差异。本轮不新增独立的数据下载器或环境安装步骤。
