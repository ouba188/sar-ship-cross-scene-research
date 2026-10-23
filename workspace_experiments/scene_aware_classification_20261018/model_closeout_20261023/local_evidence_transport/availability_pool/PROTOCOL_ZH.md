# 固定参考可用性诊断

仅源端、零训练、零新选点、零骨干提取。代码只写本目录，不改父文件，不Git。外部189仅为问题来源，本轮不读取外部缓存、图像、标签或成绩。

范围：stem/layer1，pooled、independent、anchored及其body_match。加载已冻结PCA、已保存的每查询每类两条refs，不重新检索、不补第三条参考。用现有sinkhorn/pair_score重算原始逐参考输出，先重放父缓存的mean(2)。

结构可用性由原有效极化标记和区域正像元质量决定，不看余弦、标签、OT匹配量或neff阈值。pooled/independent需query和reference相应区域存在；anchored的side/end还需双方body存在。body_match只启用body，其他列为设计屏蔽，不计入缺失统计。

每对参考输出为v_k=(sum T_ij sim_ij, sum T_ij)，结构标记a_k。旧读出为(v_1+v_2)/2；新观察量为sum(a_k v_k)/sum(a_k)，无可用参考时输出0并保存有效标记/计数。**绝不除以OT匹配量，不把语义拒配改成结构缺失。** 固定refs及每参考运输不变；一个实际可用但被拒配的参考仍占一个分母名额。

全源角色仅生成无标签结构覆盖与缓存，留待授权后匹配训练；本轮不评价selection/evaluation预测。主要标签诊断仅utility_train644条：逐真类、极化、区域，记录自身候选类及固定seed17 full基底最高非真类的结构支持和相似度/匹配量margin变化。该margin属于输入描述，不是最终分类logit，也不是精度估计。seed17仅固定竞争类，不以多个seed重复计样本。

先完成诊断和建议，不训练，不挑修复力度，不把算术缺失处理当核心创新。后续若授权训练，应原mean与valid-reference mean使用相同refs/架构/初始状态/预算及各自utility_train-only scaler；不能将旧head直接换输入的结果冒称公平训练对照。
