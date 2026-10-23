# 几何覆盖与设计记录

2026-10-23。CPU尺寸/footprint准备已完成，没有提取网络特征或训练。初版讨论的保度随机图仅为未训练设计记录；最终授权实验见PROTOCOL_ZH.md，完整图取代随机null。

源fold0共1821图：train1542，validation279。source_moments的ID、reader与原fold0 train/validation精确一致；train用各自inner held读出，validation用outer，沿用fit/held MMSI及pass隔离。47种原尺寸；3642条极化覆盖记录（两极化几何相同，不能当3642独立对象）。

|每极化|n|有效格数中位|Kish有效格中位|合法四邻接边中位|Kish<4对象|无邻边|
|---|---:|---:|---:|---:|---:|---:|
|train|1542|25|16.9905|40|211|0|
|validation|279|36|21.0785|60|27|0|

最少4格/4边；train有效格范围4-196，validation4-169。边缘fractional格的权重质量占比中位train39.08%、validation37.57%，不能二值化。边权Kish只是权重集中度，不是独立边样本量，邻边共享端点且深层RF大量重叠。两组数据的支持不同，不能把比例统计自动当作去除尺寸影响。

实际看了既有000560（77px）与000621（65px）双极化原图/类证据图，以及canvas三例。000560的横向亮结构与000621几个横向亮区提供可检验的排列动机；相邻类证据格也有同号连续或正负切换。图仅rotation0粗格，不能证明该连续性具有额外类别效用，更不能定位具体物理机制。原边缘矩与同位置类对矩都对全部类别向量和权重共同置换不变，而邻边能量可改变，因此存在数学上未被旧统计保留的信息。

prepare_coverage.py生成coverage.csv、coverage_summary.csv、geometries.csv、summary.json。null_graphs.json为一次确定性同度组slot排列；每节点无权度/边数保持，但乘积边权重算，edge mass比可到0.7807，加权度变化并非零。这一null并非完美物理置乱，最终不采用、不训练。最终改为解析完整图，更清楚地对照局部邻接与幅度/有效数。

Geary类比和已有工作边界由主流程MECHANISM_NOTE_ZH.md记录；本支线不声称首创新统计。初期signed统计提议不采用。准备阶段无GPU、无test计算、无训练、无Git。
