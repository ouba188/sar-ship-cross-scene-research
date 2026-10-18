# F_0921 clean PNG 库盘点

2026-10-18。用户提供路径 `E:/Hermes/paper_tracking/f0921_chips_clean_png`。已完成目录/索引配对检查、一个样本双极化像素重放和两例VV/VH实际图像查看；未训练、修改数据或解压大型NPY。

## 已核实

- 类别目录共有498148张PNG，即249074对VV/VH；224×224灰度uint8的存储格式由导出脚本、NPY头部和各类抽查图像共同确认，未逐图全库解码。
- 九个目录为旧八类加OTHER。全部文件名配对无缺失，全部行号的类别归属与ZIP内index.csv对应，无遗漏/多余行号。
- ZIP索引包含21970个MMSI、6256个日期×UTM分组、99种class_raw字符串。后两者不是6256个唯一卫星产品或99个正式分类标签。
- `f0921_chips_clean_npy.zip` 中仍有249074行的两个NPY与42.64MB index.csv。只读取ZIP目录、索引和各极化第一行，不需要解压约25GB数组。
- 第一行VV/VH PNG像素与ZIP内NPY逐像素一致。这只验证该对导出，不称全库像素复核。

| 目录 | 成对裁片 |
| --- | ---: |
| bulk_carrier | 33958 |
| container_ship | 34382 |
| crude_oil_tanker | 19532 |
| fishing_vessel | 6053 |
| general_cargo | 14024 |
| offshore_supply | 15099 |
| product_chemical_tanker | 36156 |
| tug_towing | 38372 |
| OTHER | 51498 |

## 对应关系

`E:/Hermes/scripts/cron_png_guard.py` 中历史命令明确从 `f0921_chips_clean` 调用 `export_chips_png.py`，导出到用户指定目录。导出直接将NPY转PNG，没有再次拉伸；来源仍是已重采样/拉伸后的动态裁片，不是原始SAR定标值。

PNG名字中的row是clean子集行号，必须首先连接ZIP内index.csv的row，再通过row_src查原动态行号。例如container_ship的row32对应动态row38，不是动态row32。索引还保留scene、row_in_shp、MMSI、class_raw、crop_side_src等。类目录只代表粗分类，不能从OTHER推断其没有细标签。

本次检查时 `f0921_chips_clean` 展开目录和此前的 `f0921_chips_dyn` 目录均不在原路径。此前动态库盘点记录保留为历史，不推断目录缺失原因；当前使用PNG及旁边的clean ZIP元数据即可继续有界分析。

## clean 的实际含义

`E:/Hermes/scripts/subset_clean.py` 的筛选是：缓存到陆地距离大于动态裁片半对角线（按10m源采样计算），再排除VV标准差小于默认1.0的裁片。它按索引复制图像，添加row_src，不重新裁片。

这不等同于人工确认没有陆地、临船、虚影或标签歧义。其几何保证依赖距离缓存及坐标对应正确；本轮没有重验全库距离。它也没有逐图检验VH是否平坦，因此不能因为无缺失文件就说两极化内容都有效。

筛选会改变靠岸/离岸及类别组成；clean库适合现有图像阅读、类别分析和相应任务研究，但不能作为港岸及复杂邻域的唯一代表数据。不要把它与全池用不同查询直接比较出所谓模块增益。

## 图像查看

- clean row0 / 动态row0 / MMSI477427600：VV/VH都显示中央近水平延长亮响应及周围纹理，图像内容可直接读取。没有据外观验证其船型标签。
- clean row32 / 动态row38 / MMSI366576000：VV/VH中央亮响应外有明显交叉延伸；不把该结构直接命名为尾迹、邻船或某个物理散射机制。

两对图像均按原存储224尺寸查看，未再拉伸。索引与每类抽查路径见sample_paths.csv。已打开的是上述两对，不是该文件里所有抽查图都已人工视觉复核。

后续优先从PNG直接调取实际查询/支持图；元数据从ZIP索引读取。需要更大邻域、原尺度、物理散射或整景关系时，再回到可追溯原景，不从PNG反推不存在的信息。不会为读取PNG另行整库解压或下载。
