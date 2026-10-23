# 源OOF梯度机制检查

只诊断OSS2已完成local/additive的6fold×seed17/29/43，共36个已选头；每头检查按原seed重建的零末层初始化及保存的selected checkpoint，两状态共72次。均用同fold已保存T(A OOF)、inner mean/moments和该头保存的源scaler，不使用validation/test标签或输入，不更新任何参数。

原jointCE=0.5 mean_i CE_i+0.5 mean_c mean_{i:y=c} CE_i。因此单样本梯度为g_i=[0.5/N+0.5/(8 n_y)] grad_theta CE_i。原头没有额外MMSI权重，不把reader Ridge的MMSI权重迁入本诊断。全部参数梯度平方范数e_i=||g_i||²，贡献占比e_i/sum_j e_j；此占比不是对||sum_i g_i||²的可加分解，也不是AdamW实际更新占比。

CPU torch.func.grad/vmap分块计算；验证各单样本加权loss之和与原joint_loss一致，各g_i之和重放原整批joint_loss参数梯度。零末层初始化须精确输出B；selected沿用已保存epoch和state，无重选点。保存逐样本、类别、最大贡献个体、005904及检查回执。若005904不在fold源训练，标为不在source，不能补算测试梯度或记作零贡献。

另记录整批梯度范数和有符号聚合方向投影用于解释抵消，以及additive两支参数梯度能量，不增加训练臂。只在当前两个状态判断瞬时CE梯度，不能声称还原全部200步、Adam动量、预条件器或解耦weight decay。原selection_curves中epoch e的train_loss在第e步更新前记录，本次checkpoint梯度是更新后选中状态，不能拿该loss直接作同状态数值断言。
