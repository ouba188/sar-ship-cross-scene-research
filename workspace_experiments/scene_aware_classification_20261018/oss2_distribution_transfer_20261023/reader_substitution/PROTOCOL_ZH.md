# 固定源验证读出器替换诊断

2026-10-23。目的：区分原local/additive源验证效用是否明显依赖OOF训练与outer验证的局部统计读出器变化。已有同source图换W时sigma变化，但包含该图参与outer拟合，不能单独解释验证失效。

本次只用六fold已有source validation；该fold的4个inner和outer都未训练于这些验证船/pass。保持图像、B、mu、head、descriptor scaler、选中epoch全固定，只将sigma/kappa从outer换成4个inner统计的等权平均。无单reader择优、无新训练、无新定标、无测试指标。原90头local/additive各18个均评价，asinh与新antisymmetric不混入。

这不是纯尺度实验：W方向/fit规模/集成同时改变，也未让训练和推理完全同分布。阳性仅说明当前选中头对读出器组织敏感，不证明某一个因素是根因；阴性也不能证明任何合法对齐训练都无用。源验证曾参与选点，结果仅为诊断。平均只作用分布，additive均值保持outer原值，避免把均值变化混为局部矩作用。

记录同样本配对OA/BA/NLL、救回/误伤、逐类/折、零步头以及替换的标准化幅度。固定seed17各arm四结局原图回看；若缺结局如实记录。不得据验证替换结果挑新主模型。
