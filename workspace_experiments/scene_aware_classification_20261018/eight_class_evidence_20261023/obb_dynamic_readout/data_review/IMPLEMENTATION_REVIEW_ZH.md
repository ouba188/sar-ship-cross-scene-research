# 定向实现审阅

## 当前结论

已只读审阅 `common.py`、`coverage.py`、`fit_readers.py`、`extract_regions.py` 与 `frozen_config.json`，未发现需阻塞当前源提取/训练准备的实际数学或坐标错误。不重跑覆盖、reader拟合或提图；未修改这些文件。

## 已核对的关键路径

- **坐标与C4：正确。** 角点中心按每pol的pad-crop平移，角度模pi，实际中心0/stride4的Voronoi积分包含边界半像素。每view旋转像素和mask后重新计算cell质量；不是在单张feature map上转格子冒充C4骨干前向。详见REVIEW_ZH.md的实测shape/原图。
- **固定模板来源：正确。** coverage.py按每个outer/inner reader的fit索引及各pol有效标志计算L/W中位数，不把outer模板偷偷用于inner；模板保留目标原中心、原轴向。目标动态几何与source模板的区别实际进入mask。
- **辅助reader来源与线性回放：正确。** 继承原15个fit/held范围，inner按MMSI/product/pass排除held，每fit覆盖8类。StandardScaler/PCA128/postscale仅fit；outer alpha在源validation balancedNLL选，inner复用对应outer alpha。实际W展开包含postscale和原scale及固定logit×3，intercept包含各层平移，记录的线性回放误差约1e-14。新layer1方向没有误用2048维旧layer3 W。
- **温度：正确。** 每fold/pol温度来自各inner held的global sigma、仅有效pol与8类，按C4先sigma后平均取中位数；各区域与臂共享。没有读取test标签或测试性能来确定温度。
- **区域摘要：正确。** 同一W产生各区域mu，区域内空间中心化并减类均值；两侧softmax采用log归一化权重±q/tau，每view非线性计算后除4累计。缺区域有临时数值分布，随后按固定region_valid清零，未把空区域NaN当观测。动态与fixed有效性分别计算，使用四view最小n_eff>=4与正面积。
- **共同质量接口：符合冻结方案。** summaries保留共同动态几何和动态质量，不让dynamic臂独占额外几何数值。缺pol沿用已有valid并保持零摘要，不抹掉有效VV。

## 训练与推理补审已完成

`run_heads.py`出现后已只读审阅其load_info/raw_inputs/standardize/source_train/test_evaluate，并完成固定案例回放。未发现当前实现的实际泄漏、坐标或臂接口阻塞。

- load_info源train按四个inner held索引拼接并检查每行恰一次；validation/test按对应outer加载，不混同一次观测在不同fold中的模型。
- baseline按fold/seed/role/instance_id取冻结attention分数，核对obs_id、MMSI、class；不重新校准旧路径。
- dynamic/fixed分别读取summary区域1:4与4:7，使用相同32维动态质量、相同维度及初始state。两臂各自摘要scaler来自本fold源train，共同质量scaler逐项相同；不是要求不同分布的摘要共用同一scaler。
- class-shared MLP在类轴去均值，末层零初始化；jointCE、200步、每5步source J、包含0步，精确分数严格大于才更新，保留earliest tie。test只恢复已选state与scaler，不再选epoch。
- merged是有效动态区域摘要均值后输入，不是与dynamic同容量的纯身份置换；global/merged565参数、fixed/dynamic853参数，不能宣称四臂严格同容量。

36头中12个选0；dynamic5/9、fixed4/9。seed17仅fold0 dynamic非零（80步），fixed对应190步；fold1 dynamic0、fold2两者均0。原图机制展示事前限定active优先，不以零回退拒用有害修正作为新区域图像证据。源学习曲线由主流程分析，本支线不重复训练或作根因预判。

六个指定案例共12个已选头回放，最大score误差4.77e-7（float32单例/批量量级），预测一致。区域干预严格是原始mu/h/b置0后重建中心化槽，quality/valid/B/scaler/head均不变；结果见CASE_REVIEW_ZH.md。不称重训消融或像素删除因果。
