# 原始发布与本地数据的解释边界

查阅日期：2026-10-13。依据原作者论文全文及Copernicus官方产品文档，不用后续分类论文的简化数据集介绍代替原始定义。具体本地统计见dataset_audit，不将论文发布规模直接当本实验规模。

## 原始论文

1. Huang等，OpenSARShip: A Dataset Dedicated to Sentinel-1 Ship Interpretation，IEEE JSTARS 11(1):195-208，DOI `10.1109/JSTARS.2017.2755672`。核读II节构建、III节属性、IV节组织和表I。原始发布列出11346 ship chips与41幅Sentinel-1影像；类型先AIS匹配，再以MarineTraffic细化。发布区分原始、校准、灰度增强、伪彩色，不应混用为同一种物理测量。GRD的分辨率20x22米与采样间隔10x10米不是同一个数。灰度uint8图经过增强，不是sigma0数值本身。
2. Li等，OpenSARShip 2.0: A large-volume dataset for deeper interpretation of ship targets in Sentinel-1 imagery，BIGSARDATA2017，DOI `10.1109/BIGSARDATA.2017.8124929`。核读2.1-2.4及第3节：这是扩展版而非独立生成的数据集；增加样本、干扰标签与标签层级。声明34528 chips/87产品，包含GRD和SLC，也含同一成像的两种产品。干扰区分多船和岸边。细分类并非所有样本都有，因此八类精细标签筛选会改变队列。

可复查全文入口（原作者上传ResearchGate）：publication `320390622`、`321407682`；官方发布页：`opensar.sjtu.edu.cn/DataAndCodes.html`。发布数字应与本地XML目标数、VV/VH文件数分别核对，不能把双极化文件数当独立船舶样本数。

## 官方产品定义

Copernicus SentiWiki的S1 Products及S1 Processing区分SLC斜距复数产品与GRD检测、多视、地距产品。两者不只是分辨率数字不同；几何、相位和多视也不同。将SLC导出的灰度图直接Resize到方形，并不能变成GRD等价输入。

旧论文表I给SLC采样2.3x17.4米，而现行SentiWiki表给IW SLC 2.3x14.1米。对此不猜测成因、不替本地文件填值；需要以实际产品元数据、去突发/地理编码记录及裁片变换为准。

本轮随后已在完整产品名.xml中恢复全部11个入选OSS2 SLC产品的实际间距：距离向2.329562米，方位向13.927475至14.028690米。早期仅查Metedata.xml而判断缺失的记录已更正。999个目标尺寸和22个逐像素幅度/显示探针支持当前派生图仍在原采样网格；据此完成正确轴与等面积错轴的有限处理对照。它仍只处理现有显示图，不能称重建物理分辨率、完整地理校正或统一辐射定标。实际流程与结果见FINDINGS_ZH.md第5节。

## 对本轮实验的约束

- 核对当前输入具体来自Patch_Uint8、Patch_RGB还是其他派生目录；标签字段、排除数量、真实产品数、时间、像素长宽和极化分别记录。
- 用MMSI、采集时间/轨道/datatake检查相关性，不以不同产品文件名证明独立。同船同次成像也不自动等同于像素重复。
- 同一目标的GRD/SLC配对可以减少船型和场景差异，但不是随机化物理因素试验；跨版本差异不能全部归于产品。
- 原论文不会证明我们模型失效的原因。只有本地事实加同样本干预结果，才能将候选解释提升为当前模型中的已验证原因。
