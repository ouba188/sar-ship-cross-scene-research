# M3相近类别关系上下文：已接入，尚未验证分类收益

日期：2026-11-18。用户授权将LDP最有用的差别融入现有模型。

## 本轮结果

已实现完整chip到类别输出的候选入口，默认使用相近类别加权上下文。保留旧等权模式及原权重，不覆盖已选中的模型检查点，不更改论文结果。没有重训或保存新训练权重。

本轮单元测试及两张真实chip、16个既有合法源参考的完整前向/反向验证通过。新候选新增参数为0，旧模式算子回放误差为0；骨干、M3、CCREL和PCEA梯度均非零。不是只画流程图而未接入。

## 唯一算法改动

在每个极化、每个区域内，从合法参考得到各类均值mu_c。原先：

    other_c = mean(mu_d, d != c且d有效)

现在：

    a_cd = softmax_d(cos(mu_c, mu_d)), d != c且c,d均有效
    other_c = sum_d a_cd * mu_d
    condition_c = concat(mu_c, other_c, mu_c - other_c)

仍由原有metric生成器产生有界通道度量w_c = 1 + 0.5*tanh(metric(condition_c))。温度固定1，未调参。self-attention的对角线和不可用类被严格屏蔽；无其他有效类别时上下文为0。

相似性直接由可训练的参考key原型产生，没有detach，因此更新能传回原有浅深投影及编码器。query标签不参与生成。上下文只因当前query的合法参考掩膜而可能不同，并非由该query图像的最强竞争类直接生成，不能称为逐查询错误诊断。

## 没有改动的部分

- 原20个细槽、VV/VH分路、body/side/end区域身份、原图尺度和OBB。
- M1、M2、CCREL、PCEA结构以及已有可训练状态。
- M3局部竞争项仍使用原本定义；本轮没有将加权上下文再重复加入该项。
- aggregate传输边际聚合、距离比较、回写投影及layer3之后的作用位置。
- 参考数量、参考标签及既有合法性限制；没有加入未标注目标数据。
- 没有加入coarse_plus，也没有新增频域、尺寸分组、首尾识别或正反匹配。

## 文件与使用

- relation_operator.py：RelationContextMatch及install(network, mode)。替换已有network.matcher时必须在创建优化器和CUDA graph之前调用；保留state_dict键、权重、参数requires_grad和训练状态。
- model.py：Model()默认relation，Model('uniform')为同起点等权对照。沿用当前routed_s17开发模型起点，不把它重标为论文正式最佳模型。
- verify.py：合成原型机制检查、完整旧算子回放、真实chip前向与反向检查。
- verification.json：不含样本身份的检查结果。
- integration_local.json：私有样本ID、logit和6组权重，只留本地。
- verification_inputs_local.json：本地两例验证样本选择，不公开上传；公开代码的真实chip检查需要自行提供合法数据和此配置。

底层历史CompetitiveMatch仅增加other_class_context扩展点，默认数学操作保持原样；新算法只在新子类启用。调用路径：Model -> network.matcher -> 按极化/区域形成mu -> other_class_context -> 原metric -> 原比较与回写 -> 后续骨干与分类。

由于没有新增参数，旧新state_dict形状相同。保存训练候选时必须同时记录mode='relation'或'uniform'，不能仅凭权重文件判断采用哪一种上下文。

本地运行：

```powershell
& 'D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe' 'E:/临时会话/paper_draft_20261108/m3_relation_context_20261118/verify.py' --integration
```

## 验证范围及实际响应

已检查：相近类权重大于较远类、自身权重0、不可用类别0、仅一类/无类有限输出、相似度相同时均权、类别重排等变、原型梯度、缺极化情况下完整算子有限输出、strict权重加载、旧模式精确回放。

实际两chip验证没有使用标签损失或进行optimizer.step，反向仅用logit平方均值检查可微性，不能视为训练实验。GPU峰值约3.753GiB；VV/VH各三区域共6次上下文调用。真实检查点安装前/安装后uniform的输出差为0，CPU及CUDA随机状态保持。最大logit变化7.689e-6。此次8类均有效，非对角权重为0.128854至0.147497，均权为1/7≈0.142857。当前表示下差异较弱，不通过标签试探额外放大温度。

两个既有案例的原始VV/VH及同类/混淆类参考已回看，复用上一轮gallery的03、04面板：一个散货查询比同类参考更宽、另一短小拖船查询与同类及普通货参考大小不同。它们说明仅凭尺寸或整体外观不能决定关系权重应如何调整，不证明等权上下文就是此前错误的根因。本次两例预测均未翻转，一例持续正确、一例持续错误。新权重是否能通过配套训练带来净救回尚未验证。

## 下一步应验证什么

同起点、同源训练批次/增强/步数下比较uniform与relation，联合更新原有可训练骨干和模块。按源验证选模，再比较既定开发队列，并回看救回/误伤的原chip与真实参考。若仅体现共同类别偏移而无净收益，不据注意力图宣称有效；有收益再做错配关系权重对照以定位内部机制贡献。

这份代码完成了用户要求的结构接入，不代表已经完成上述收益实验，也不自动替换已有正式模型。
