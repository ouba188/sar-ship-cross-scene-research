# SLC 几何建议的最终证据状态

2026-10-13。**早期“OSS2 SLC spacing 缺失”结论已经撤回：实际元数据使用完整产品名.xml。主线已完成有界两臂几何对照，取代此前临时收尾决定；本审计分支只独立核对已有fold 2结果，不运行几何实验。核对见GEOMETRY_IMPLEMENTATION_AUDIT_ZH.md。**

最终文件 `product_spacing_metadata.csv` 含全部 11 个入选 SLC 产品。逐样本 `oss2_slc_native_grid.csv` 含 999 行，所有 range_spacing=2.329562 m，azimuth_spacing=13.927475–14.028690 m；`srgr_flag=0`、无地形校正。

999/999 原始、定标、双极化 uint8 图像宽高相同，999/999 与 XML crop bounds 一致。ReadMe 明确 x=range、y=azimuth。11 产品各取一个 chip、两极化共 22 个探针，raw 复数幅度到同位置 uint8 的映射均单调，幅度排序后的 uint8 逆序总数为 0。支持 uint8 未预先纠正 native SLC 采样纵横比；不是全发布代码的形式化证明。

因此，以每产品精确间距和每 chip incidence 构造 `range_spacing/sin(theta)/azimuth_spacing` 的固定 range 压缩，是有实证依据的局部平面几何敏感性控制。比值范围 0.2327367–0.3272724，中位 0.2505111。**不使用 2.3/14.1 或 2.3/17.4 的常数替代本地值。**

主线实际执行的对照限定原 source、原完整 OSS2 支持集、原 uint8 条件，GRDH 不变；range 压缩+居中补框，与相同实际压缩面积的 azimuth-sham 对照。相同插值、整数尺寸规则和补边；不调参、不重拟合源模型；变换后的原支持集仍须重解原有支持适应系数。主读数R/T/DD不需要mapper。读取233个同标签同SAR时间的对应观测时，逐draw只选两端都是query的pair，并报告实际类数。该比较不能排除插值、补边、分辨率和上下文变化，因此不直接成为“物理根因已证实”。

主配对文件 `oss2_grdh_slc_pairs.csv` 为 234 对，其中 `same_label=True` 233 对；两端 query 掩码见 `oss2_grdh_slc_pair_query_eligibility.csv`。其余 370 对仅 datatake 匹配另存，不混入同时间主表。原始标签保持不变。
