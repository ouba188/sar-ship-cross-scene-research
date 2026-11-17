# 独立审阅

2026-11-18，Chandrasekhar只读审阅，未运行训练或推理、未修改文件。

## 历史暴露核查

- 203候选与当前staged fit1205/adapt136/development160/references16/expanded3075按路径、MMSI比对无命中。
- 与更早utility池fit1205/validation296/utility644/selection612/development485及身份排除145条无命中。
- 更早三折基线的实际train成员有5条命中。主代理在首次推理前将其排除，固定198条主分析。
- 其余198条的无暴露仅指已核查记录，不能称所有历史实验绝对未使用；3次过境已经参与原验证，不是独立场景。

## 实现审阅

未发现阻断问题：

- 精确加载原九个arm/seed的best.pt，strict=True，无新选点。
- Full17/29/43原选点210/240/180步；generic/class_detail为180/240/240步。
- 198查询、16参考、8回放输入分开，参考和回放不计入评价分母。
- 保留参考顺序及旧恢复、native160裁剪/补零、原几何积分和/255；新查询不擅自恢复极化。
- Full17八条数值回放误差1.27e-6，预测一致；未声称九个模型均逐个回放。
- 每个诊断条件完整重置disabled/uniform/permuted；配对方向、救回误伤、BA及分层统计正确。
- 三个种子重复不作为594个独立船只。

Full与PCEA比较包含配套训练和不同源内选点；新增类别组织结合class/generic及同权重uniform/permuted判断。路径诊断不冒充重训消融。主代理另查看全部6幅原始chip面板，图像分析见FINDINGS_ZH.md。
