# 定标和热噪声处理依据

2026-10-24读取senbox-org/s1tbx公开实现；所读取master对应提交1daae60d572e3ad0ee98c0ce3538c61ad71d7fa1（2023-04-26）。不是声称这是最新SNAP版本。

1. `s1tbx-op-calibration/src/main/java/org/esa/s1tbx/calibration/gpf/calibrators/Sentinel1Calibrator.java`：约第622–654行的LUT双线性插值和幅度平方后除LUT平方；第741–758行的applyCalibration提供相同公式。
2. 同目录上一级`gpf/Sentinel1RemoveThermalNoiseOp.java`：约第681–706行对幅度平方减噪声功率，负值置trgFloorValue=1e-5；第1202–1236、1311–1339行将距离噪声插值与方位噪声插值相乘；第1355–1414行说明距离噪声向量按方位块时间范围选择。

固定版本代码地址：

- https://github.com/senbox-org/s1tbx/blob/1daae60d572e3ad0ee98c0ce3538c61ad71d7fa1/s1tbx-op-calibration/src/main/java/org/esa/s1tbx/calibration/gpf/calibrators/Sentinel1Calibrator.java
- https://github.com/senbox-org/s1tbx/blob/1daae60d572e3ad0ee98c0ce3538c61ad71d7fa1/s1tbx-op-calibration/src/main/java/org/esa/s1tbx/calibration/gpf/Sentinel1RemoveThermalNoiseOp.java

本轮仅针对两个实际GRD产品做局部实现：三块IW方位噪声矩形无重叠且覆盖全部使用像素，每块覆盖全部27条距离噪声向量的行范围，因此不用SNAP通用流程中的空向量回退。程序对这些实际前提作了检查，不假装支持其他布局。

设原DN为D，sigmaNought标定幅度表为A，噪声功率为N，则本轮输出：

- sigma_amp = D/A；
- sigma_noise_amp = sqrt(max(D²-N,1e-5))/A；
- sigma_power = D²/A²，单独作为非线性表示对照。

用SciPy RegularGridInterpolator双线性读取A和距离噪声，方位噪声单独线性读取。104个局部窗口重新从measurement取值，逐像素与第14轮缓存完全一致；实际LUT节点再现误差阈值1e-10，全部通过。未运行完整SNAP独立逐像素复验。尝试GDAL SAFE的SIGMA0子数据集接口返回Measurement bands not found，因此没有把该接口当作独立验证结果。

这些是幅度/功率图像，不是SLC相干相位；未做地形校正、斑点滤波、运动补偿。热噪声去除也不等于海杂波去除。
