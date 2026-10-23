# OSS2 GRDH方法复验：只做可行性与固定划分

2026-10-23。当前阶段不训练、不下载、不操作Git；只写本目录。复用已有public_sar审计与metadata，代码按karpathy-guidelines约束为有明确判别用途的数据处理。

## 预先固定的划分

1. 从指定target_manifest保留scene_id含`_GRDH_`的全部2746条，维持原manifest行序，不按interference、尺寸、类别难度筛除。八类顺序沿用0..7，索引为本地cohort.csv零基row_index，同时保留原manifest位置与target_manifest_row/sample_id。
2. pass为平台+绝对轨道+datatake，不按产品片段假定独立采集。先将27个pass字典序排列，`numpy.random.default_rng(20261023).permutation`仅一次，再`numpy.array_split(...,6)`连续分为六包。不以类别平衡或任何模型分数调整分包。
3. outer fold k的test为包k，候选validation为包(k+1)%6，候选train为其余四包。validation排除test全部MMSI；train排除候选validation和test全部MMSI。test保留全部，任何被排除行保留在membership并标原因。
4. outer已合法train内，以剩余四个原pass包分别作为inner held；该inner fit为其它三个包的outer train，再排除held全部MMSI。每个outer train行恰一次inner held；各fit与对应held身份、pass均互斥。
5. 先统计每个outer/inner fit与validation/test/held的八类n、MMSI、pass；没有模型指标。若fit缺类，不补类别、不搜seed、不悄悄改折数；保留初始失败方案，并根据实际支持分布提出更少inner或外折的单一合理方案，另行明确状态。

## 图像与独立性核查

全队列只核对两路PNG文件头尺寸/模式与manifest；复用public_metadata的head/tail及目标中心。最多每类一例VV/VH原图，初始拟看本类最大native窗口。文件头/现成几何检查发现1×1及端点越界后，在尚未查看案例像素前改为：优先本类退化尺寸，其次端点未落入范围，再最大native窗口，最后sample_id升序。这是针对实际数据风险的选择，不涉及模型表现。只为这八例从对应XML读取目标crop extent核对坐标，非全量XML重审；不代表典型性能。

160中心裁窗须区分“截去PNG”“截去已标head/tail”“是否完整保留船体”三件事；head/tail不自动构成有宽度的OBB。若现有几何不足以保证船体不裁，优先保留全PNG、统一224窗口居中补零（以实际最大尺寸核对后定），不resize、不删除大船；真实像元footprint参与加权池化，padding不是新增有效观测。

与24港全manifest及当前15251 cohort对照采集年份、产品、pass、MMSI；年份/产品不交不等于从未开发过OSS2。明确既往公开实验暴露，当前任务为新分组协议下的方法可迁移性复验，不是全新盲测。

## 后续模型边界（本轮不执行）

冻结的是additive方法结构及既定预算，不宣称24港权重零样本迁移。OSS2基线为本地合法源R50 weighted GAP→源内预处理/Ridge→源校准T；辅助W、均值、局部矩和OOF必须来自相应合法fit。它不含24港oldA+D完整模型，也不含新邻域检测。后续训练须独立授权，并明确T是否二级OOF；不能把基础OOF冒称全流水线OOF。
