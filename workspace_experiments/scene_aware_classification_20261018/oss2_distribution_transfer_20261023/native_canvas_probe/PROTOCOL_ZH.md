# Native canvas固定敏感性诊断

2026-10-23，运行前锁定。仅fold0 source validation279，冻结outer W/intercept、ImageNet R50及seed17 calibration。无训练、无test推理/评价，无选尺寸、阈值或模型。

## 唯一改变

复用extract_features.native_image生成224原输入，再在四边各补32个同值零像素成为288。原PNG像素完整保留，无resize/crop/插值图像。零填充发生在ImageNet归一化之前，与原实现一致。C4、VV/VH、fp32/eval、layer3、area footprint权重及归一化池化原样复用；stride16，新增边界对应每边2格。288的18x18网格中央14x14与224对齐，C4各向均核对footprint相等、外围pool权重为0。

不修改父脚本globals。网络内部卷积padding不改；这里仅改变输入黑色画布边界远近。footprint排除无效位置直接池化，不等于排除卷积RF传播。观察到变化也不能自动归因为某个物理机制或认定288是更好部署方案。

## 固定输出与检查

先核对224重提特征与原features，复算A=feature W^T+intercept，并经原seed17 calibration得到B。浮点检查feature rtol1e-4/atol5e-5，A/B绝对误差1e-4以内；报告实际误差/top1。若失败先定位，不以放宽误差掩盖。同一模型计算288，保存逐例特征差幅、A/B完整分数/flip、OA/BA/NLL/Brier、类别与原图长边固定分层(<=64,65-128,129-192,193-224)。这些分层仅描述，不选点。

保留小型源特征及C4逆转对齐后的channel-mean activation和channel-RMS差图，不保存全量F。固定B224到B288四结局首ID，若缺失明确记录；实际查看每例VV/VH原图和粗网格变化图。图像不缩放进入模型；展示网格不当成精细物理定位，RF跨越footprint边界。原图源路径、类别、预测和分差同时存表。

参考arxiv1811.11718仅作为相关已有工作的标识；主流程报告已读取PDF前4页，不是本支线独立阅读，官方实现未读。本轮不实施partial convolution，不作padding方法首创主张。完成固定两尺寸诊断即停止。
