# 共同倍率作用位置解耦：单候选36头

2026-10-23，运行前固定。old/new各6fold×3seed，36头；同一三类5695目标、已冻结q/oq/env/base/oof和划分，无重拟合，无新readout、calibration head、温度搜索或weighted-neutral臂，只写本目录，不Git。

网络仍79->32->32、3616参数和末层零初始化。有效组按selected_kind，layer3前16零槽不参与均值。a=.5tanh(net)，m=有效组a均值，d=a-m；新score=(1+m)*(A+Σd*q)。共同倍率位于完整判别分数之后，不再通过不等b输出-m*b类别偏移；m和d仍共享同一有界a，并非独立自由参数。

训练forward显式传入env[train], oq[train], oof[train]；源val/test用env、outer q和outer base。训练CE保持原float32精度和类别/MMSI权重；评估用保存的原base精度，保证0步完全等于A。200step、seed17/29/43、AdamW(.003,.01)，每5步以完整新score的源val balanced NLL选步、含0步、改善阈值1e-6。未缩放core仅诊断，不用于选模型或改训练目标。

验证全部样本的新score和A+contrast argmax一致；1+m>0，d有效组和≈0，零槽精确零。无独立环境类别偏置：A和q均零时score零；仅q零时是同一A的正倍率，不错误要求score-A为零。该倍率可改变NLL/Brier，不改变固定d下argmax。

保存OA/BA/NLL/类别平均NLL、Brier/类别平均Brier、class、救回误伤、逐例分数与未缩放core、PT、曲线、来源、时间、独立回放。NLL为自然对数，Brier为每样本三类概率平方误差之和再平均，不除以类别数。复用已完成original/contrast/range作同容量不同算子参照，不重训。

每5步记录当前训练子集m均值/分位、d跨组SD、有效组|a|>.49比例；选中模型分别汇报train/validation/test及测试逐例。这用于检查正共同饱和是否伴随差异容量缩小，不预设其发生，不加新臂。理论上m接近.5时d可变化空间会缩小，下一次先分logits再分别有界化仅可能的后续假设，当前不执行。

原图按每队列seed29新score救回/误伤/CC/WW首ID、优先无邻域；原尺度VV/VH复用，无插值。新旧仍是重复开发划分，不独立新数据确认。主流程文献说明通道竞争/温度等已有方法，本轮候选是针对共同倍率作用位置的可证伪改动，不称首创。允许负结果、不追加搜索。
