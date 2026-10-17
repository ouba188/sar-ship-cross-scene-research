# OpenSARShip 1.0 联合几何元数据准备

## 本轮输出与范围

只准备元数据，不训练、不估计船轴、不改既有输入或划分。当前八分类队列是 GRDH，不是所提供的 SLC 功能坐标表。geometry.csv 严格保留现有 target_manifest.csv 的行序。

```json
{
  "rows": 1349,
  "products": 22,
  "ships": 1110,
  "classes": 8,
  "radar_axis_valid": 1349,
  "incidence_valid": 1349,
  "paired_native_available": 1349,
  "raw_available": 1349,
  "calibrated_available": 1349,
  "xml_crop_extent_matches": 1349,
  "slc_rows": 2553,
  "slc_products": 18,
  "exact_slc_product_overlap": 0,
  "incidence_min_deg": 30.768978,
  "incidence_max_deg": 45.778946,
  "export_probe_count": 44,
  "export_probe_identity_best": 44,
  "export_probe_min_identity_rho": 0.8900719101912788,
  "cache_ids_exact": true
}
```

## 主代理接入约定

- `global_row` 是当前 1349 行 manifest / oss1_inputs128.npy 的零基行号；`target_manifest_row` 是历史全量标签映射的行号，不可当作本缓存行号。
- `sample_id` 按产品 + 原生中心 x/y 对回 Ship.xml，同时逐行检查 XML 序号、MMSI、细分类标签及 VV/VH 文件中心；没有按相似船型或文件遍历顺序拼接。
- `chip_path_vv/vh` 是模型使用的未 resize Uint8 TIFF；`raw_patch_path` 是原始幅度双通道，`calibrated_patch_path` 是标定双通道。ReadMe 定义两者通道 0=VH、1=VV。
- `radar_range_axis_deg`、`radar_azimuth_axis_deg` 从图像向右为 0 度，向下为 90 度，是模 180 度的无向轴。有效时分别为 0、90。联合 YOLO 船轴时必须使用同一图像坐标。
- 主读出接口：`radar_axis_rad` 为上述距离轴的弧度值，`radar_valid` 为有效性；`incidence_rad` 为入射角弧度值，`incidence_valid` 独立标记。无效角留空，不填伪零。
- `incidence_deg` 是对应 GRDH 样本中心的入射角，来自 Ship.xml 的 SAR 字段，不是平面旋转角；有效性独立于雷达轴与后续船轴质量。
- 不输出 AIS 航向或人工船轴；MMSI 只用于身份和分组，class_id/name 只用于标签。
- `radar_direction_sign_valid=False`，有向视线和地理方位字段留空。右视、升降轨不能直接变成裁片里的有向视线。当前只确认横向距离轴、纵向方位轴。
- 本队列雷达无向轴与图像坐标轴重合，`radar_axis_rad` 全为 0 是真实栅格语义，不是各船都有不同雷达平面角。相对于检测船轴的夹角仍随船变化，入射角也随样本变化。因此不能把单独拼接常数 0 的效果解释成新增观测信息；联合局部组织应与图像坐标读出公平比较，区分坐标重新表达与入射角条件信息。

## 直接依据与有限验证

1. 每个产品随附 ReadMe.pdf，正文第 2 页图 2 标注 x(range)、y(azimuth)；第 7 页把 Chip_Direction 定义为船相对于方位向的角度，North_Direction 是船相对于北的方向。因此两字段均未用作雷达观测角。
2. 22 个实际产品的 Metedata.xml/Abstracted_Metadata 均为 GRD/DETECTED，map_projection='-'、is_terrain_corrected=0、srgr_flag=1；range_spacing=10m，azimuth_spacing 逐产品保留真实值，不把采样间隔说成分辨率。
3. 逐张读取 TIFF 尺寸与 Orientation，并将显示裁片、原始幅度裁片和 XML 包围尺寸对照。产品中第一条 manifest 样本的 VV/VH，另用幅度与 Uint8 的像素秩相关检查 8 种转置/翻转/旋转，这是每产品代表样本的导出方向核对，不是全部 2698 张图的逐像素重建。
方向核查比较 identity 是否优于其他方向，不用绝对亮度秩相关阈值作为几何质量门槛；部分 Uint8 显示增强会降低原始幅度与显示值的相关程度，这不等于方向不明。
4. public_expansion/cache_oss1.py 仅灰度读取和双线性 resize，未旋转或翻转；缓存 oss1_ids.npy 与 manifest sample_id 全序列精确一致。下游新增旋转、翻转或转置增强时，必须同步变换雷达轴。

## 不跨产品移植的内容

提供的 SLC 表有 2553 行、18 个产品，与本轮 GRDH 产品精确交集为 0。相同采集的 SLC/GRD 也不是同一像素网格，不能移植其图像方向、中心坐标或按行拼接入射角。
该表和生成脚本中的 orientation_relative_azimuth_deg、func_slc_axis_image_deg、func_sar_ship_axis_geo_deg 均未进入本输出。

## 还未恢复的量及可用路径

当前无向轴足够表达船轴与距离向/方位向的相对轴角，但不能区分朝雷达一侧与背雷达一侧。若后续需要有向地面视线，原始 Metedata.xml 已含 Orbit_State_Vectors、first/last_line_time、line_time_interval、latitude/longitude/slant_range_time 网格以及右视标志；应确认裁片像素到原始栅格的有向映射，再用轨道与目标定位求解，并作图核对。未在本轮推算或伪造。
注意：XML UpperLeft/LowerRight 的数值命名可能与数组左右大小相反，本轮只用绝对跨度验证裁片尺寸，不据此推断 near/far 的正负。

## 四折覆盖

保持 public_full_20261014/run.py 的 pilot + 3 expanded 划分，不合并各折训练角色。support_draws_B16 的 rows 包含五次抽样重复，unique_samples 才是该折支持样本数。

| fold              | role              |   rows |   unique_samples |   radar_axis_valid |   incidence_valid |
|:------------------|:------------------|-------:|-----------------:|-------------------:|------------------:|
| pilot             | source_train      |    720 |              720 |                720 |               720 |
| pilot             | source_validation |    202 |              202 |                202 |               202 |
| pilot             | query             |     84 |               84 |                 84 |                84 |
| pilot             | support_draws_B16 |    160 |               85 |                 85 |                85 |
| S1A_010685_00FED5 | source_train      |    790 |              790 |                790 |               790 |
| S1A_010685_00FED5 | source_validation |    210 |              210 |                210 |               210 |
| S1A_010685_00FED5 | query             |     38 |               38 |                 38 |                38 |
| S1A_010685_00FED5 | support_draws_B16 |    160 |               93 |                 93 |                93 |
| S1A_011246_011040 | source_train      |    815 |              815 |                815 |               815 |
| S1A_011246_011040 | source_validation |    234 |              234 |                234 |               234 |
| S1A_011246_011040 | query             |     37 |               37 |                 37 |                37 |
| S1A_011246_011040 | support_draws_B16 |    160 |               64 |                 64 |                64 |
| S1A_011421_0115FD | source_train      |    886 |              886 |                886 |               886 |
| S1A_011421_0115FD | source_validation |    227 |              227 |                227 |               227 |
| S1A_011421_0115FD | query             |     40 |               40 |                 40 |                40 |
| S1A_011421_0115FD | support_draws_B16 |    160 |               70 |                 70 |                70 |

## 类别数量

|   class_id | class_name              |   samples |
|-----------:|:------------------------|----------:|
|          0 | bulk_carrier            |       816 |
|          1 | fishing_vessel          |        35 |
|          2 | general_cargo           |       121 |
|          3 | product_chemical_tanker |       116 |
|          4 | container_ship          |       218 |
|          5 | crude_oil_tanker        |        13 |
|          6 | tug_towing              |        18 |
|          7 | offshore_supply         |        12 |

## 导出方向探针

| scene_id                                                            | sample_id                                                                         | polarization   | best_transform   |   identity_rho |   best_nonidentity_rho |
|:--------------------------------------------------------------------|:----------------------------------------------------------------------------------|:---------------|:-----------------|---------------:|-----------------------:|
| S1A_IW_GRDH_1SDV_20150823T221233_20150823T221258_007396_00A2CA_54D0 | S1A_IW_GRDH_1SDV_20150823T221233_20150823T221258_007396_00A2CA_54D0:x1673:y6973   | vv             | identity         |       0.994563 |              0.180793  |
| S1A_IW_GRDH_1SDV_20150823T221233_20150823T221258_007396_00A2CA_54D0 | S1A_IW_GRDH_1SDV_20150823T221233_20150823T221258_007396_00A2CA_54D0:x1673:y6973   | vh             | identity         |       0.987614 |              0.173974  |
| S1A_IW_GRDH_1SDV_20150916T221233_20150916T221258_007746_00AC44_BA63 | S1A_IW_GRDH_1SDV_20150916T221233_20150916T221258_007746_00AC44_BA63:x1866:y6375   | vv             | identity         |       0.973101 |              0.275149  |
| S1A_IW_GRDH_1SDV_20150916T221233_20150916T221258_007746_00AC44_BA63 | S1A_IW_GRDH_1SDV_20150916T221233_20150916T221258_007746_00AC44_BA63:x1866:y6375   | vh             | identity         |       0.995521 |              0.243574  |
| S1A_IW_GRDH_1SDV_20150928T221239_20150928T221304_007921_00B100_32A3 | S1A_IW_GRDH_1SDV_20150928T221239_20150928T221304_007921_00B100_32A3:x1945:y2475   | vv             | identity         |       0.998986 |              0.0993725 |
| S1A_IW_GRDH_1SDV_20150928T221239_20150928T221304_007921_00B100_32A3 | S1A_IW_GRDH_1SDV_20150928T221239_20150928T221304_007921_00B100_32A3:x1945:y2475   | vh             | identity         |       0.996182 |              0.106019  |
| S1A_IW_GRDH_1SDV_20151010T221234_20151010T221259_008096_00B5A1_5C25 | S1A_IW_GRDH_1SDV_20151010T221234_20151010T221259_008096_00B5A1_5C25:x2023:y6034   | vv             | identity         |       0.983099 |              0.122621  |
| S1A_IW_GRDH_1SDV_20151010T221234_20151010T221259_008096_00B5A1_5C25 | S1A_IW_GRDH_1SDV_20151010T221234_20151010T221259_008096_00B5A1_5C25:x2023:y6034   | vh             | identity         |       0.99662  |              0.0785178 |
| S1A_IW_GRDH_1SDV_20160102T221226_20160102T221251_009321_00D788_F5BA | S1A_IW_GRDH_1SDV_20160102T221226_20160102T221251_009321_00D788_F5BA:x1959:y9609   | vv             | identity         |       0.993682 |              0.101562  |
| S1A_IW_GRDH_1SDV_20160102T221226_20160102T221251_009321_00D788_F5BA | S1A_IW_GRDH_1SDV_20160102T221226_20160102T221251_009321_00D788_F5BA:x1959:y9609   | vh             | identity         |       0.997258 |              0.190491  |
| S1A_IW_GRDH_1SDV_20160114T221226_20160114T221251_009496_00DC88_46B2 | S1A_IW_GRDH_1SDV_20160114T221226_20160114T221251_009496_00DC88_46B2:x2597:y10134  | vv             | identity         |       0.993115 |              0.274892  |
| S1A_IW_GRDH_1SDV_20160114T221226_20160114T221251_009496_00DC88_46B2 | S1A_IW_GRDH_1SDV_20160114T221226_20160114T221251_009496_00DC88_46B2:x2597:y10134  | vh             | identity         |       0.994442 |              0.313994  |
| S1A_IW_GRDH_1SDV_20160126T221225_20160126T221250_009671_00E1AF_228F | S1A_IW_GRDH_1SDV_20160126T221225_20160126T221250_009671_00E1AF_228F:x1171:y10117  | vv             | identity         |       0.99406  |              0.253212  |
| S1A_IW_GRDH_1SDV_20160126T221225_20160126T221250_009671_00E1AF_228F | S1A_IW_GRDH_1SDV_20160126T221225_20160126T221250_009671_00E1AF_228F:x1171:y10117  | vh             | identity         |       0.995096 |              0.284144  |
| S1A_IW_GRDH_1SDV_20160219T221225_20160219T221250_010021_00EBE1_C9D5 | S1A_IW_GRDH_1SDV_20160219T221225_20160219T221250_010021_00EBE1_C9D5:x7957:y14018  | vv             | identity         |       0.937157 |              0.195417  |
| S1A_IW_GRDH_1SDV_20160219T221225_20160219T221250_010021_00EBE1_C9D5 | S1A_IW_GRDH_1SDV_20160219T221225_20160219T221250_010021_00EBE1_C9D5:x7957:y14018  | vh             | identity         |       0.986268 |              0.142692  |
| S1A_IW_GRDH_1SDV_20160302T221225_20160302T221250_010196_00F0D2_D0C8 | S1A_IW_GRDH_1SDV_20160302T221225_20160302T221250_010196_00F0D2_D0C8:x1652:y9818   | vv             | identity         |       0.987158 |              0.289592  |
| S1A_IW_GRDH_1SDV_20160302T221225_20160302T221250_010196_00F0D2_D0C8 | S1A_IW_GRDH_1SDV_20160302T221225_20160302T221250_010196_00F0D2_D0C8:x1652:y9818   | vh             | identity         |       0.98803  |              0.16502   |
| S1A_IW_GRDH_1SDV_20160326T221226_20160326T221251_010546_00FAC9_4D44 | S1A_IW_GRDH_1SDV_20160326T221226_20160326T221251_010546_00FAC9_4D44:x10876:y14542 | vv             | identity         |       0.998493 |              0.281696  |
| S1A_IW_GRDH_1SDV_20160326T221226_20160326T221251_010546_00FAC9_4D44 | S1A_IW_GRDH_1SDV_20160326T221226_20160326T221251_010546_00FAC9_4D44:x10876:y14542 | vh             | identity         |       0.998332 |              0.17361   |
| S1A_IW_GRDH_1SDV_20160328T095421_20160328T095446_010568_00FB64_215C | S1A_IW_GRDH_1SDV_20160328T095421_20160328T095446_010568_00FB64_215C:x5998:y8270   | vv             | identity         |       0.991301 |              0.15778   |
| S1A_IW_GRDH_1SDV_20160328T095421_20160328T095446_010568_00FB64_215C | S1A_IW_GRDH_1SDV_20160328T095421_20160328T095446_010568_00FB64_215C:x5998:y8270   | vh             | identity         |       0.983032 |              0.133096  |
| S1A_IW_GRDH_1SDV_20160405T102459_20160405T102528_010685_00FED5_97D5 | S1A_IW_GRDH_1SDV_20160405T102459_20160405T102528_010685_00FED5_97D5:x394:y2746    | vv             | identity         |       0.993693 |              0.099841  |
| S1A_IW_GRDH_1SDV_20160405T102459_20160405T102528_010685_00FED5_97D5 | S1A_IW_GRDH_1SDV_20160405T102459_20160405T102528_010685_00FED5_97D5:x394:y2746    | vh             | identity         |       0.992238 |              0.113061  |
| S1A_IW_GRDH_1SDV_20160407T221226_20160407T221251_010721_00FFEA_ADBD | S1A_IW_GRDH_1SDV_20160407T221226_20160407T221251_010721_00FFEA_ADBD:x1734:y10348  | vv             | identity         |       0.990115 |              0.176606  |
| S1A_IW_GRDH_1SDV_20160407T221226_20160407T221251_010721_00FFEA_ADBD | S1A_IW_GRDH_1SDV_20160407T221226_20160407T221251_010721_00FFEA_ADBD:x1734:y10348  | vh             | identity         |       0.997175 |              0.135152  |
| S1A_IW_GRDH_1SDV_20160419T221227_20160419T221252_010896_01052F_69E2 | S1A_IW_GRDH_1SDV_20160419T221227_20160419T221252_010896_01052F_69E2:x1155:y10113  | vv             | identity         |       0.997145 |              0.25132   |
| S1A_IW_GRDH_1SDV_20160419T221227_20160419T221252_010896_01052F_69E2 | S1A_IW_GRDH_1SDV_20160419T221227_20160419T221252_010896_01052F_69E2:x1155:y10113  | vh             | identity         |       0.995841 |              0.330435  |
| S1A_IW_GRDH_1SDV_20160513T221231_20160513T221256_011246_011040_617A | S1A_IW_GRDH_1SDV_20160513T221231_20160513T221256_011246_011040_617A:x2774:y7587   | vv             | identity         |       0.992196 |              0.166941  |
| S1A_IW_GRDH_1SDV_20160513T221231_20160513T221256_011246_011040_617A | S1A_IW_GRDH_1SDV_20160513T221231_20160513T221256_011246_011040_617A:x2774:y7587   | vh             | identity         |       0.97074  |              0.221903  |
| S1A_IW_GRDH_1SDV_20160523T102501_20160523T102530_011385_0114CC_4FED | S1A_IW_GRDH_1SDV_20160523T102501_20160523T102530_011385_0114CC_4FED:x508:y5345    | vv             | identity         |       0.997733 |              0.365572  |
| S1A_IW_GRDH_1SDV_20160523T102501_20160523T102530_011385_0114CC_4FED | S1A_IW_GRDH_1SDV_20160523T102501_20160523T102530_011385_0114CC_4FED:x508:y5345    | vh             | identity         |       1        |              0.286721  |
| S1A_IW_GRDH_1SDV_20160525T221231_20160525T221256_011421_0115FD_6054 | S1A_IW_GRDH_1SDV_20160525T221231_20160525T221256_011421_0115FD_6054:x1759:y8386   | vv             | identity         |       0.997503 |              0.125667  |
| S1A_IW_GRDH_1SDV_20160525T221231_20160525T221256_011421_0115FD_6054 | S1A_IW_GRDH_1SDV_20160525T221231_20160525T221256_011421_0115FD_6054:x1759:y8386   | vh             | identity         |       0.996768 |              0.176609  |
| S1A_IW_GRDH_1SDV_20160606T221232_20160606T221257_011596_011B8A_E6B1 | S1A_IW_GRDH_1SDV_20160606T221232_20160606T221257_011596_011B8A_E6B1:x2019:y7943   | vv             | identity         |       0.996899 |              0.137793  |
| S1A_IW_GRDH_1SDV_20160606T221232_20160606T221257_011596_011B8A_E6B1 | S1A_IW_GRDH_1SDV_20160606T221232_20160606T221257_011596_011B8A_E6B1:x2019:y7943   | vh             | identity         |       0.993982 |              0.162318  |
| S1A_IW_GRDH_1SDV_20160630T221233_20160630T221258_011946_012696_1AA1 | S1A_IW_GRDH_1SDV_20160630T221233_20160630T221258_011946_012696_1AA1:x1773:y7918   | vv             | identity         |       0.996284 |              0.225579  |
| S1A_IW_GRDH_1SDV_20160630T221233_20160630T221258_011946_012696_1AA1 | S1A_IW_GRDH_1SDV_20160630T221233_20160630T221258_011946_012696_1AA1:x1773:y7918   | vh             | identity         |       0.993716 |              0.205617  |
| S1A_IW_GRDH_1SDV_20160712T221240_20160712T221305_012121_012C5B_8569 | S1A_IW_GRDH_1SDV_20160712T221240_20160712T221305_012121_012C5B_8569:x1607:y4188   | vv             | identity         |       0.994691 |              0.169892  |
| S1A_IW_GRDH_1SDV_20160712T221240_20160712T221305_012121_012C5B_8569 | S1A_IW_GRDH_1SDV_20160712T221240_20160712T221305_012121_012C5B_8569:x1607:y4188   | vh             | identity         |       0.995561 |              0.192204  |
| S1A_IW_GRDH_1SDV_20160817T221236_20160817T221301_012646_013D9A_4929 | S1A_IW_GRDH_1SDV_20160817T221236_20160817T221301_012646_013D9A_4929:x1424:y8034   | vv             | identity         |       0.994031 |              0.33343   |
| S1A_IW_GRDH_1SDV_20160817T221236_20160817T221301_012646_013D9A_4929 | S1A_IW_GRDH_1SDV_20160817T221236_20160817T221301_012646_013D9A_4929:x1424:y8034   | vh             | identity         |       0.996214 |              0.283178  |
| S1A_IW_GRDH_1SDV_20161018T095407_20161018T095432_013543_015ACD_AB15 | S1A_IW_GRDH_1SDV_20161018T095407_20161018T095432_013543_015ACD_AB15:x15674:y12118 | vv             | identity         |       0.890072 |              0.282024  |
| S1A_IW_GRDH_1SDV_20161018T095407_20161018T095432_013543_015ACD_AB15 | S1A_IW_GRDH_1SDV_20161018T095407_20161018T095432_013543_015ACD_AB15:x15674:y12118 | vh             | identity         |       0.985851 |              0.169121  |

## 可复现来源

- Manifest: `E:/Hermes/paper_tracking/sensitivity/opensarship1_transfer_20261007/target_manifest.csv`
- Script: `E:/临时会话/oss1_joint_geometry_20261017/prepare_geometry.py`
- SLC reference (not joined): `E:/临时会话/slc_functional_coordinates_four_regions/opensarship1_slc_functional_coordinates_no_ais_v1.csv`
- Dataset documentation: `D:/Documents/project_root/OpenSARShip_1/S1A_IW_GRDH_1SDV_20150823T221233_20150823T221258_007396_00A2CA_54D0/ReadMe.pdf`
- Existing cache: `E:/临时会话/scene_cross_dataset_redesign_20261014/public_expansion/cache_oss1.py`
