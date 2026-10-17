# 几何元数据覆盖与坐标说明

2026-10-16；仅组装既有无标签几何，不训练、不按目标表现挑选。

## 读取接口

CSV按原manifest顺序覆盖全部样本。`global_row`和`manifest_row`为0起始行号；`sample_id`与`instance_id`相同。

`geometry[N,7]`列顺序：`ship_axis_rad, radar_axis_rad, incidence_rad, ship_valid, radar_valid, incidence_valid, ship_confidence`。
缺失角度为NaN，valid为0，缺失ship_confidence为0；加载后须先保留mask再将NaN置0。0不是缺失角度的物理含义。三个valid分别标记可用性，不表示统计独立。

## 坐标和可解释边界

- 两个axis均是无向轴，范围[0,pi)，以图像向右为0、向下为pi/2；不是船首方向。incidence是[0,pi/2]弧度，不是图像旋转角。
- 船轴复用既有VV/VH接受门限：各自亮散射二阶矩、elongation>=2、中心偏移和边界筛选，双视图地理轴差<=15度；不新增按标签选择的阈值。成对地理轴映射到VV图像作为主坐标，每极化的映射和直接测量轴均保留作诊断。
- ship_confidence=(r_min^2-1)/(r_min^2+1)，r_min是双视图最小细长比。它是协方差各向异性强度，不是正确方向概率；双极化角差单列。
- 雷达轴复用relative_conditions.range_axis_deg的绝对地理方位，不用relative_signed_deg推算。来源为同产品注释网格固定line、增加pixel的地距轴代理，既有覆盖内/最近网格<=20km限制保留。不是精确雷达波束方位；nearest_grid_distance_m只是质量诊断，不伪造置信概率。
- 将WGS84方位在样本处构造对称100米局部弦，投影到源EPSG；对GeoTIFF完整2x2仿射矩阵求逆，再按真实PNG宽高转换到等边resize坐标。该变换包含投影子午线收敛、y符号及非等比缩放；不假定北向图像、不把y向下当北向。
- 原生PNG来自轴对齐窗口裁片，未旋转。裁切/像素中心平移不影响位移轴；128x128双线性resize及当前浅层读取后续112x112等比resize不改变本表方向。若训练另加旋转/翻转或非等比缩放，必须同步变换这两个轴，不能原样使用。
- VV为共同token网格参考；保留VV/VH映射差。这只是方向，不声称两个裁片逐像素精配准。物理船轴仍是亮散射主体代理，不是人工船体标注。
- 未扩大旧几何提取范围：雷达轴缓存原来只针对通过船轴筛选者计算。因此雷达缺失部分来自历史筛选，不等于这些图像从物理上无法得到雷达方向。

## 溯源文件

- manifest: `E:\临时会话\visual_reliable_baseline\artifacts\eight_class_adaptive_20260916\manifest.csv`
- 三个方向缓存: `E:\Hermes\paper_tracking\sensitivity\relative_observation_geometry_20261009` 内view_axes.csv / paired_axes.csv / relative_conditions.csv。
- 源仿射: `E:\SAR_AIS_T_shipchip_dataset\v5_context_dynamic_64_128_return_20260915\metadata\local_knowledge_sync_v1\private` 内server*_georef.jsonl，仅status=ok且源TIFF路径精确匹配。
- 入射角: `E:\Hermes\paper_tracking\pixel_correction_review\chip_incidence_v1\incidence_by_instance.csv`，status=ok且有限值；不依赖observations的已知MMSI子集。
- 裁片: `v5_server_toolkit_20260912/scripts/build_v5_context_chips.py` 的Image.crop；合并: `visual_reliable_baseline/merge_returned_v5_dataset.py` 保留crop_left/top与源路径。
- 轴提取/方向: `E:/Hermes/paper_tracking/sensitivity/oof_reliability_check/extract_ship_axis_proxy.py`、`ship_axis_proxy.py`、`relative_radar_geometry.py`。

## 覆盖

| 范围 | 不同样本 | 船只 | 采集 | 船轴 | 雷达轴 | 入射角 | 三项齐全 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 全manifest | 38091 | 18766 | 841 | 4640 | 4228 | 28628 | 4228 |
| Antwerp-Bruges | 886 | - | - | 0 | 0 | 380 | 0 |
| Busan | 871 | - | - | 0 | 0 | 871 | 0 |
| Callao | 476 | - | - | 42 | 41 | 252 | 41 |
| Fujairah | 1158 | - | - | 0 | 0 | 909 | 0 |
| Hamburg | 1619 | - | - | 0 | 0 | 520 | 0 |
| Houston | 804 | - | - | 90 | 90 | 487 | 90 |
| Jebel Ali | 4602 | - | - | 1180 | 1085 | 3749 | 1085 |
| Los Angeles | 583 | - | - | 124 | 124 | 522 | 124 |
| Melbourne | 64 | - | - | 0 | 0 | 0 | 0 |
| Mombasa | 149 | - | - | 9 | 9 | 50 | 9 |
| New York | 470 | - | - | 137 | 134 | 395 | 134 |
| Newcastle | 299 | - | - | 24 | 21 | 153 | 21 |
| Ningbo-Zhoushan | 5756 | - | - | 0 | 0 | 5711 | 0 |
| Port Hedland | 859 | - | - | 59 | 41 | 300 | 41 |
| Port Klang | 3425 | - | - | 988 | 936 | 3285 | 936 |
| Port Said | 1895 | - | - | 154 | 115 | 1135 | 115 |
| Qingdao | 1981 | - | - | 0 | 0 | 622 | 0 |
| Richards Bay | 744 | - | - | 43 | 43 | 492 | 43 |
| Rotterdam | 573 | - | - | 0 | 0 | 136 | 0 |
| Santos | 1658 | - | - | 390 | 390 | 1343 | 390 |
| Shanghai | 2390 | - | - | 0 | 0 | 1544 | 0 |
| Singapore | 5259 | - | - | 1171 | 975 | 4754 | 975 |
| Sydney Botany | 304 | - | - | 15 | 15 | 212 | 15 |
| Tanger Med | 1266 | - | - | 214 | 209 | 806 | 209 |

## 五池实际B32源episode覆盖

按冻结query_coverage的B32任务汇总唯一支持/查询样本，不把重复episode暴露算成独立证据。每池独立列出，不能合池训练；以下不使用目标标签。

| 池/阶段/角色 | 不同样本 | 船只 | 采集 | 船轴 | 雷达轴 | 入射角 | 三项齐全 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Jebel Ali/fit/s | 572 | 537 | 10 | 135 | 135 | 505 | 135 |
| Jebel Ali/fit/q | 705 | 649 | 15 | 145 | 145 | 577 | 145 |
| Jebel Ali/selection/s | 93 | 92 | 2 | 0 | 0 | 24 | 0 |
| Jebel Ali/selection/q | 42 | 42 | 2 | 0 | 0 | 42 | 0 |
| Ningbo-Zhoushan/fit/s | 484 | 478 | 10 | 51 | 51 | 224 | 51 |
| Ningbo-Zhoushan/fit/q | 467 | 455 | 17 | 26 | 22 | 218 | 22 |
| Ningbo-Zhoushan/selection/s | 234 | 231 | 4 | 46 | 46 | 194 | 46 |
| Ningbo-Zhoushan/selection/q | 294 | 284 | 5 | 53 | 53 | 226 | 53 |
| Qingdao/fit/s | 593 | 588 | 10 | 97 | 97 | 490 | 97 |
| Qingdao/fit/q | 623 | 618 | 11 | 108 | 108 | 475 | 108 |
| Qingdao/selection/s | 231 | 227 | 2 | 49 | 22 | 231 | 22 |
| Qingdao/selection/q | 623 | 576 | 2 | 176 | 91 | 623 | 91 |
| Shanghai/fit/s | 619 | 595 | 10 | 64 | 53 | 358 | 53 |
| Shanghai/fit/q | 706 | 660 | 18 | 16 | 5 | 363 | 5 |
| Shanghai/selection/s | 167 | 152 | 4 | 30 | 30 | 81 | 30 |
| Shanghai/selection/q | 164 | 154 | 4 | 28 | 28 | 80 | 28 |
| Singapore/fit/s | 281 | 279 | 7 | 45 | 45 | 166 | 45 |
| Singapore/fit/q | 279 | 276 | 10 | 49 | 49 | 208 | 49 |
| Singapore/selection/s | 201 | 201 | 4 | 25 | 25 | 130 | 25 |
| Singapore/selection/q | 168 | 166 | 6 | 0 | 0 | 71 | 0 |

### 对本轮设计的直接限制

- Shanghai池fit查询只有5/706例双轴齐全；这限制了几何交互的实际监督，不能用训练episode重复次数代替不同有效样本量。
- Jebel Ali池selection支持/查询均无双轴；Singapore池selection查询无双轴。这些选择集仍可评价完整分类与缺失回退，但不能单独检验查询几何交互。
- Ningbo-Zhoushan、Qingdao、Shanghai港的现有船轴均为0覆盖；零覆盖是旧可精确关联的返回数据/筛选范围限制，不是这三个港没有物理方向。在这些目标港，C与B/A的差异不能直接归因于使用了可靠双轴。
- 因此完整队列结果与双轴可靠子集结果必须分开解释。本文件不自行扩展轴提取、不修改原训练划分，也不根据该限制挑目标港。

## 已运行的针对性验证

- 原像素轴 -> 既有地理轴 -> 本次像素轴回放，最大误差0.00003056度。阈值0.01度用于发现符号/仿射/缩放错误，不用于筛选模型收益。
- VV/VH雷达方向映射差：中位数0.00000261度，最大0.00493124度。
- 全manifest行序、sample_id和源episode global_row匹配，缺失值与valid一致；入射角与轴均检查合法范围。
- 不读取预测结果、不调整目标抽样、不启动训练、不下载大景。覆盖是数据可用性，不是分类效果证据。
