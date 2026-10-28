# 复现与作用位置

## 运行

沿用本机ClearSAR环境，工作目录为本目录：

```powershell
& 'D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe' -X utf8 -B run.py --mode prepare
& 'D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe' -X utf8 -B run.py --mode train
& 'D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe' -X utf8 -B diagnose.py
& 'D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe' -X utf8 -B run.py --mode cost
& 'D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe' -X utf8 -B analyze.py
```

可选算子位于 `sar_m123_standalone/sar_m123/block_m2_placement.py`，`Layer2BlockM2(state, 'preadd', block_index=1/2/3)`。默认block_index=3不变，旧position调用不改行为。当前发布默认模型不替换；Git归档对应current包下同一可选文件。

对选中block：conv1/BN/ReLU -> conv2/BN/ReLU -> conv3/BN -> M2 -> identity相加 -> ReLU。再正常执行该stage剩余block，然后进入原layer3、M1/M3及读出。M2在VV/VH分别处理，共享骨干参数；layer2卷积不更新，但下游运算保留梯度，layer3.5三个卷积继续按旧协议微调。

训练缓存仅在新增M2之前：block1使用prefix2_1，block2使用prefix2_2，block3沿用prefix2。旧raw2和anchor3仍为M3比较保护锚点；不能把旧raw2当新的后续值特征。16条参考和查询都经过相同活跃路径。缓存由已有原生crop/pad输入生成，无插值缩放或重标注。

依赖前轮shallow_m2_writeback的缓存、m2_block_placement前缀及runner、m2_residual_placement已有九个对照结果和checkpoint、current发布包teacher及manifests。新六次模型保存本目录checkpoints。`prepare`使用训练源4例检查非零操作与梯度，检查模型随后销毁；实际训练重新从统一teacher及M2初始化开始。

冻结部分、保存checkpoint重放等沿用旧训练器。`diagnose.py`固定seed17，按结局组内最大绝对分差变化选例，另补相对末段的分歧例，不根据这些案例选参数。`off_*`同时关闭查询及参考的对应新M2路径，不改原图，不重新拟合，解释为模型依赖而非物理因果。

成本脚本从已有stage1及保护锚点开始，各臂都重新执行所需的整个layer2值路径，16查询与16参考、4次warmup、20次计时。baseline/stage不能直接用已缓存raw2省略该段。只比较共同入口下的局部延迟，不作为从chip到分类的端到端成本；训练计时因缓存位置不同也不能直接作为部署加速结论。

公开归档不包含权重、原始数据、完整私有manifest或特征/预测缓存。保留代码、小型结果表、诊断与派生原图面板；依赖资产缺失时不可声称下载仓库即可重现训练。
