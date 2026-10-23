# 冻结source-pilot链外部覆盖确认

2026-10-23。只运行已冻结权重推理，不训练、不拟合PCA/scaler/tau、不重新选点，不做局部匹配或本轮分类头推理。主流程另行负责这些步骤。

实际索引纠正：fold0/1/2 test分别1561/1632/1551，三折不重复并集才是4744。fold0不是4744。目标与当前upstream全部四个源角色比较，MMSI重叠1669条、过境重叠0条。依用户预授权，排除任一源角色身份或过境重叠，保留3075条，含fold0全部1561条、fold1 756条、fold2 758条。evaluation角色虽非训练，也保守排除以避免已开发源查询身份再现。保存完整资格及排除表，不改变旧标签/索引。

使用regional_utility_pilot/upstream/models的axis_ridge、five、neighborhood、calibration、attention、body、full及两个reader W，使用原scales.json tau。A/D输入来自全局OLD既有缓存，以obs_id映射，绝不使用旧全量主模型权重替代当前source-pilot权重。

原生160裁补、不缩放。每极化每C4同一骨干前向同时生成旧链mean3/hb3/summary1/quality/geometry及20槽stem64/layer1 256描述。旧链粗区域>=4门槛保留；20细槽记录neff、不作>=4筛选。真实footprint与缺失极化按原接口保留。

先抽固定源行及既有220/1128/150/11556案例重放：检查冻结A/D/T、摘要、tokens及三策略。结构字段/布尔标记要求一致，几何质量误差<=1e-8，GPU摘要与tokens<=1e-4，最终策略<=5e-5且top1一致；这是运行前固定数值容差，不按目标成绩调节。重放通过才处理外部队列。另核对新layer3 GAP与原全局冻结特征及W线性闭合。

只输出标签覆盖统计，不进行外部选模；该历史队列经历过开发，且本轮确认见源结果后追加，不称盲测。输出table保留ROOT全局row_index，所有数组顺序一致；保存原始4744资格表与剔除原因。无Git、无父文件改动。
