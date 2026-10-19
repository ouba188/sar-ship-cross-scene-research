# 有边界的 Sentinel-1 轨道几何恢复

2026-10-19 完成。`orbit_geometry.csv` 已交付并冻结，不再改数值。本支线仅负责元数据几何，没有读取图像像素、运行分类或提交 Git。

## 结果与失败原因

- 原 cohort 269 行，保持原 row 顺序、clean_row 和 observation_key 对齐。
- 132 行有效，恰好覆盖旧 geometry_valid 的 132 行，涉及 16 个产品；没有扩大旧有效集合。
- 137 行保持未知：108 行旧缓存仅有 incidence、没有可用原生轴；16 行同时存在这种缺失及多产品归属；13 行多产品归属。它们是沿用旧门槛的排除，不是轨道插值失败。
- 有效候选没有下载、产品匹配或时间覆盖失败。复用 3 份本地完整 annotation，补下载 13 份公开小 XML，共 22,683,810 字节；没有整景下载或凭据读取。
- 对旧无向 grid range axis 的最小模 π 角差：中位数 0.0851869°，均值 0.0882008°，P95 0.1266371°，最大 0.1772773°。
- 轨道 incidence 减同位置 grid incidence：中位数 +0.0292326°，范围 +0.0244705° 至 +0.0362328°。两种计算本来就不是完全相同的近似，差值不应直接当成已验证的轨道误差。

详细产品计数、路径、下载 URL、状态向量数量和覆盖范围保存在 `orbit_receipt.json`。原始补下载文件仅保存在 `E:/Hermes/paper_tracking/sensitivity/f0921_orbit_metadata_20261019/`。

## 计算与坐标约定

1. 继承 `axis_radar_readout_20261019/geometry.csv` 的 `geometry_valid AND pixel_product_unique AND native_window_matches_manifest_raster`。整块可能贡献窗口必须归属同一产品；不重新读图，不凭新轨道推断混合块归属。
2. 从原始 annotation 的 `adsHeader` 核对平台、首末时间、绝对轨道号、datatake 和 VV 极化。公开镜像 URL 取自既有 `download_inventory.csv`。旧压缩 grid 与新 annotation 的坐标、line/pixel、时间、incidence/height 逐项一致后复用旧 grid。
3. 将网格经纬度投影到当前 raster 的实际 UTM CRS，在包含目标位置的三角形内对 `geolocationGridPoint/azimuthTime` 做重心插值。以局部时间差运算，避免直接加权大 Unix epoch。不以产品起始时刻替代每位置时间，也不从列表次序或升降轨猜方向。
4. 使用 `generalAnnotation/orbitList/orbit` 的 Earth Fixed 位置和速度，按插值时刻进行三次 Hermite 插值。速度取同一位置插值函数的导数。思想复用 `all_factor_probe_20261017/metadata/recover_metadata.py` 中 ECEF/ENU、Hermite 逻辑，本实现独立，不导入旧脚本副作用。
5. 目标为 WGS84 椭球高 h=0 的近海平面代理。ECEF 向量严格为 `sensor_position - target_position`，转到目标处 ENU，保留三维单位 LOS 与三维 Earth Fixed flight velocity 的 ENU 分量。入射角为 `atan2(norm(LOS_EN), LOS_U)`。
6. 用目标处正负 1 m 的东、北测地线差分得到 ENU→UTM Jacobian，再用当前 raster 二维 affine 的逆矩阵转换为图像方向，保留 UTM 子午线收敛角。图像 x 向右，y 向下；这里的栅格为 UTM 东向正、北向负。方形 chip 的均匀缩放不改变角度。

有向角 `los_angle_image_rad = atan2(image_dy,image_dx) mod 2π`，范围 `[0,2π)`；`los_axis_image_rad` 为前者 mod π。所谓“有符号”是保留射线指向，不是要求角值小于零。`flight_angle_image_rad` 同样保留有向飞行矢量。`axis_difference_deg` 是 LOS 水平投影与旧 range 轴的最小无向角差，范围 `[0,90]`，不能用于判断射线反向。

## 接口

| 字段 | 含义 |
|---|---|
| `row`, `clean_row`, `observation_key` | 冻结 cohort 的稳定对齐键 |
| `orbit_valid` | 归属、网格、实际时间及轨道全部有效 |
| `los_angle_image_rad` | target→sensor 的图像有向水平角 |
| `los_axis_image_rad` | 无向轴，mod π |
| `orbit_incidence_deg` | 相对目标处椭球法线的入射角，单位度 |
| `azimuth_time_utc` | 逐位置 grid 时间插值结果 |
| `annotation_path`, `grid_path` | 实际使用的原始文件 |
| `axis_difference_deg` | 对旧 grid axis 的无向最小角差 |
| `los_*_unit`, `flight_*_mps` | ENU 三维 LOS 单位向量、飞行速度 |
| `los_map_*_unit`, `los_image_*_unit` | 归一化二维 LOS 水平投影 |
| `flight_map_*_unit`, `flight_image_*_unit` | 归一化二维飞行方向 |
| `orbit_status`, `old_geometry_status` | 本轮结果及原门槛原因 |

未知行全部新增几何数值为 NaN，不填默认方位。有效行 `los_image_x_unit/y_unit = cos/sin(los_angle_image_rad)`；使用它们再乘 `sin(incidence)` 得到所需水平分量。`los_east_unit/north_unit` 则已经包含三维单位 LOS 的水平幅值，不应再重复乘 sin。

## 时间覆盖与近似边界

`adsHeader` 首末时间与 `imageInformation/productFirstLineUtcTime/productLastLineUtcTime` 一致；网格边缘时间与首末行存在原始亚毫秒差异，仅此边界比较允许最多一个 XML 原生行时间间隔。目标插值时刻严格位于实际首末行区间，轨道同时覆盖整个采集区间与目标时刻；禁止外推。

132 个有效目标距离采集起点最少 1.393909 s，距离终点最少 0.410327 s。三维单位 LOS 与归一化 Earth Fixed 飞行速度点积的绝对值最大 1.9664e-6，与本轮数值的近零 Doppler 几何一致；这只是内部一致性证据，不是独立精度认证。

海平面用 h=0 椭球面近似，不等于大地水准面、潮位或实际船体高度。保存插值 grid height 便于追溯，但没有据此改目标高度。轨道来源为产品 annotation，未独立获取或验证精密轨道，不能称为精轨。有限网格和局部投影微分不等于完整地形 range-Doppler 求解，也不提供散射机制或分类增益证明。

## 来源阅读范围

实现直接依据本地原始 XML 的实际字段和已有 ECEF/ENU/Hermite 代码。本支线 `web.run` 搜索及打开返回空，未声称自行在线读到规范。

本轮主代理也尝试了以下官方入口，但工具未返回可核查正文，不能记录为本轮已精读来源。以下仅为后续规范核查入口：

- [S1 Products](https://sentiwiki.copernicus.eu/web/s1-products)。
- [S1 Processing](https://sentiwiki.copernicus.eu/web/s1-processing)。
- [Document Library](https://sentiwiki.copernicus.eu/web/document-library)。

规范PDF本轮未读，不记录版本号或声称核实了条文。因此不将geolocationGridPoint的zero-Doppler、line/pixel中心语义写成“本轮已从PDF核实”。实现依据实际匹配原始XML中的坐标、时间、Earth Fixed位置速度字段及其数值验证，不添加未经来源证实的半像素偏移。主/子代理沟通中曾把官方入口描述为已读，归档前纠正，未以该描述作为实验结果证据。

## 验证与限定接入审阅

在已有 ClearSAR 环境运行恢复一次并成功完成，`python -B test_orbit.py` 的 8 项测试全部通过：ENU 解析基、target→sensor 符号及入射角、图像轴与旋转、UTM 收敛、逐位置时间插值及空间越界、解析三次运动的 Hermite 位置/速度及时间越界、未知传播、269 行输出对齐与有效范围。首次启动有一处列名列表语法遗漏，在任何恢复/下载前已修复。主代理随后实际运行主模块与几何联合测试，12 项全部通过。

对 `spatial_readout.py` 与 `run_experiment.py` 及其直接调用的池化/描述函数完成一次只读限定审阅，未发现会改变本轮结论的接入错误：

- `spatial_readout.py:98` 共用有效行；orbit 使用有向 LOS，image 使用 +x，但保留同船轴、尺寸、incidence；未知掩膜为零。
- `run_experiment.py:25` 完整保留旧核 `.5/.25/.25`，第 39 行统一加 `.0625` 新核。pooled/image/orbit 共用有效行、支持与先验。
- 支持中心化后的描述仍乘有效掩码和支持计数门槛，未知行保持零；缺有效支持时新核为零。pooled 的 8-token 重复经过同样归一化，不因重复本身四倍放大核。
- 未知查询的新核连接为零，但支持系数重解可以通过保留的旧连接改变其结果；协议已明确这一点，不应把这种变化解释为未知行获得了雷达信息。

未重跑分类、扩大审阅或修改主代码；原图和支持分析由主代理完成。几何有效 132 与主读出有效 129 的门槛不同，后者还要求可用船轴/尺寸，不能混作几何恢复失败。
