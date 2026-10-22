# 几何替代解释的小验证

固定三个输入组及超参数，先运行后解释。不改变 reference、expanded 或 strong reader 的任何文件，不追加 reference 消融，不 Git。

- body_geometry：自然对数的 `target_length_m`、`target_width_m` 及两者比值。物理尺度来自匹配的 VV 检测 OBB，不是 AIS 船体测量。
- canvas_geometry：自然对数的原生 VV/VH 窗宽、高及 `target_occupancy_vv/vh`，共六项。读取已验证的实际窗口，并与 `w_vv/h_vv/w_vh/h_vh` 核对。
- both：以上九项拼接。当前窗均为正方形，长宽比也由长宽导出，冗余项按指定保留，不做搜索。
- occupancy 沿用冻结表中原目标框面积/该极化原窗面积，运行时核对其计算关系。它不是本轮匹配 OBB 的 fg-mask 占比，不混淆这两种口径。没有显式输入角度，但框面积/占比和裁剪过程可能间接携带姿态及检测流程的关联信息；诊断不声称去除了这些关联。
- 分类器固定 `HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=7,l2_regularization=1,random_state=17,early_stopping=False)`，其余使用当前环境默认值。每折每组拟合一次，总计 18 次；不调参、不根据测试选择组。
- 直接读取 expanded 的六份 fold inputs 的 train/validation/test 索引，核对 `splits.csv` 和 strong 的 fold base 缓存完全相同。三角色 MMSI 与采集组互斥。复用既有 `run_ports24.weights`，先类别等权，再类别内每 MMSI 等权；仅 train 标签参与拟合。
- validation 仅输出预测/指标，不用于早停、调参、选组或再拟合。每个目标恰有一次测试预测，pooled 仅合并 test。保留全部三个组，不挑最佳几何组作为新主结果。
- 对照 stem 与每折 sourceval 已选 strongbase 的原缓存分数，核对已保存的 baseline_predictions。比较逐例正确/错误、强读出救回/误伤覆盖；不比较 HGB 概率与 ridge 分数的绝对尺度。
- 输出每类尺寸分布，统计以目标为单位并同时列 MMSI 数；重复船只并非独立证据。原图案例按预测关系分组，在组内选择长度最靠近中位数的案例，平手按 ID；它们是事后描述，不能替代总体比例。
- 原图直接使用 expanded/native_targets 冻结 VV/VH，保持像素值和原生尺寸；面板统一 3 倍 nearest 展示，附原 OBB 定位，不重提特征或整景解码。

问题是固定几何模型能否复现大部分分类表现及具体强读出收益，不是证明深层模型使用或未使用尺寸。几何表现低也受该固定分类器能力限制；几何表现高不等于深层表征仅靠尺寸。该队列为开发诊断，非新独立确认集。

强表征下外部 context 的现有结果按“OA 小降、BA 仍小正”保留，不能称为全面退化。
