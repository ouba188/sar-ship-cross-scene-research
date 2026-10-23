# OSS2 GRDH八类方法复验可行性

2026-10-23。可行性阶段完成，无训练、下载、Git或父文件修改。结论：**固定seed20261023的pass六折及源内四包可用，全部30个计划reader fit集合八类齐全，无需改变折数或重抽seed。建议统一224原尺度居中补零、保留所有现有PNG，不裁切、不缩放、不删大船。** 主流程随后已采用224方案并开始另行授权的模型阶段；本文不冒充模型结果报告。

## 数据与独立性

指定target_manifest的3745条中，按GRDH产品类型取全部2746条，27产品/27pass、2240个MMSI；SLC不混入，保留276条interference=1。各类n/MMSI/pass：bulk590/509/27，fishing352/338/20，general231/207/26，chemical611/456/26，container770/566/27，crude118/104/22，tug48/37/18，offshore26/23/14。没有按尺寸、几何质量或后续模型表现筛除。

OSS2实际采集时间2015-06-12至2016-07-03（2015年750条，2016年1996条）。24港全manifest38091条、24港，以及当前八类15251条、21港，均为2025/2026年：与OSS2采集年份、完整产品、平台+orbit+datatake均无交集。当前测试18港与全cohort21港不是同一分母。

但MMSI并非全新：与24港全manifest共享368个MMSI，涉及OSS2的434行；与当前15251 cohort共享190个，涉及OSS2的226行。年代隔离不是身份隔离证明，且MMSI不是独立认证的永恒实体ID。此次源拟合只用合法OSS2训练包，不调用24港已监督权重替代本地训练。

更重要的是，OSS2已经用于2026-10-07 transfer和更早公开开发；指定清单2746条全部来自此前3745条队列。旧记录含目标64支持、行级support/query未完全按身份隔离、不同encoder/队列等设置，不能把旧准确率直接当本轮基线。**当前是冻结方法、新pass/MMSI划分下的跨数据来源可迁移性复验，不是未接触的盲测、独立新数据确认，也不是24港oldAD完整模型的外部复现。** 详见`overlap_summary.csv`、`shared_mmsi.csv`和既有TRANSFER_FINDINGS。

## 固定划分与八类覆盖

27pass排序后以numpy default_rng(20261023)单次permutation，array_split为5/5/5/4/4/4个pass。test包k、val包k+1，其余train；val排test MMSI，train排val候选与test MMSI。源内四个剩余pass包逐包held，fit再排held MMSI。没有多seed选择、类别补样或标签路由。

|fold|train|validation|test|train身份排除|val身份排除|
|---:|---:|---:|---:|---:|---:|
|0|1542|279|613|257|55|
|1|1735|398|334|222|57|
|2|1571|443|455|234|43|
|3|1516|458|486|255|31|
|4|1654|333|489|234|36|
|5|1459|572|369|305|41|

六个outer train/val/test均八类齐；24个inner fit亦全部八类齐，fit大小860–1323。各类跨inner fit最小样本数分别181/49/94/167/216/23/5/5；outer train最小分别318/132/133/278/370/46/18/10。这满足八类读出可拟合条件，不代表小类估计稳定。

三个inner held缺tug（fold0 inner0、fold4 inner1、fold5 inner1），但对应fit仍八类齐。这不妨碍构建八列OOF预测；若计算这些held片段的BA，必须明确缺类分母，不能填造该类表现。每个outer train行恰一次inner held，fit/held与outer train/val/test身份互斥，pass也互斥。

小类选择方差需保留：fold1 validation tug仅1例，fold3 validation offshore仅1例；test fold2 tug仅1例，fold4 offshore仅1例。建议后续固定所有fold、汇总完整测试分母，同时报告逐类及pass/MMSI条件区间，缺类重采样明确处理；不要为稳定或涨点换seed。此处没有触发缺类fit，**不提出多余的更少折方案**。

已检查落盘NPZ的sample_ids与cohort顺序一致，六个test集合恰一次覆盖2746行。`split_checks.csv`记录身份排除及held覆盖，`fit_failures.csv`为空表（不是未运行）。

## 原尺度与目标完整性

实际读取5492张PNG文件头，两路全部存在、L模式，尺寸与manifest全部一致；本轮未解码全量像素。尺寸1–205，19例超过160（bulk1、container18），最坏160保留面积为(160/205)^2约60.92%。**截掉PNG不自动意味着裁掉船体。** 现成表的head/tail是人工端点且有方向歧义，不是有宽度OBB；缺少完整轮廓时，不能凭端点在窗内便认证整船完整。

优先复用`public_metadata.csv`的中心/head/tail；全量相对坐标以center减半幅估计并在表中明确标为estimate。8个选例仅按中心定位其XML记录，核对真实crop min/max，8例原点误差均0、crop尺寸一致，未进行全量XML重审。除两条源数据异常外，估计head/tail均在160拟议窗内；19个大窗均无已知端点被160切掉的证据，因此不把它们说成19条已裁船。

224是覆盖最大205的32倍数窗口；居中pad即可保留全部原像元，footprint只标原PNG有效区域，padding不能参与统计候选。224改变特征网格/边界条件，但不改变目标像素尺度，Ridge/T/所有对照必须共用此输入。本方案保证**不新增裁切**，不能修复源PNG已丢失的目标、伪造OBB或消除深层RF对padding/背景的混合。

两个实际问题保留在队列并标记：
- `pub2_opensarship2_006393`（bulk，test包5）VV/VH均实际1×1，唯一像元255；XML crop也1×1，head/tail均在chip外。这是没有可辨船体空间证据的发布记录，不是本轮160处理造成。224补零不能恢复信息，不应把后续响应解释为船体分布；本轮不擅自删除、改label或补图。
- `pub2_opensarship2_006067`（chemical，test包1）29×29，head相对y=-1，亮体接触上边界，XML核对确认。这是原有chip/端点完整性缺口，不能归因224，也不能声称保全原PNG等于完整船体。

## 八类各一例实际查看

已实际查看`panels/class0.png`至`class7.png`，每类仅一个对象、VV/VH并排，共8对象16个极化视图。先按几何风险选异常，再按最大窗口、sample_id排序；不涉及预测或测试择例。黄色仅是XML端点连线，红虚线仅是假设160裁窗，不标为检测OBB。完整身份、图路径、XML来源与尺寸见`review_cases.csv`。

|类/ID后缀|native尺寸|可见事实与限制|
|---|---|---|
|bulk 006393|1×1|两路只有白色单像元，无可判读船形；保留异常而不挑另一张好看的bulk图|
|fishing 000568|125×125|中心近纵向亮响应，VH沿纵向多个亮段更明显；周围还有弱响应，不能据此认证多船或物理成因|
|general 000653|93×93|中心斜向较宽亮区，两路亮度分布不同，端点均在窗内|
|chemical 006067|29×29|亮体触上边，VH周围纹理较明显；源端点在y=-1，不能认证整船已保留|
|container 005904|205×205|中心细长近纵向亮响应及横向弱条纹，红160框仍包住已标端点与主要亮区；外圈会被160删去但未证明外圈是船体|
|crude 003590|157×157|斜向亮响应较不均匀，VV亮区偏局部、VH分段更明显；不能给这些亮区赋予物理部件名称|
|tug 003847|105×105|中心多个连近亮区，周围有其他弱响应；只描述形态，不据此更改tug标签|
|offshore 005312|65×65|中心近纵向亮条，VV背景明显较亮、VH较暗；该单例不代表类别总体统计|

这里没有模型预测，故不是救回/误伤四结局分析，也不编造图像对应性能机制。异质原尺寸和小类覆盖是真实迁移条件，需带入后续结果解释。

## 交接与协议边界

- `cohort.csv`：2746行稳定本地row_index；保留manifest_position（输入文件0基位置）、target_manifest_row（上游编号）、sample_id及原路径。
- `pass_assignment.csv`：唯一一次随机排列与六包归属；`membership.csv`：每outer fold所有行的train/validation/test或身份排除原因；`inner_membership.csv`：各inner fit/held/身份排除。
- `indices.npz`：`sample_ids,row_index`及`fold{k}_{train,validation,test,train_candidate,validation_candidate}`；训练必须使用train而不是candidate。各inner键为`fold{k}_inner{j}_{fit,held,excluded_mmsi}`，全部索引指向本地cohort，不是原manifest位置。inner j按剩余包号升序。
- `split_coverage.csv`：每集合每类n/MMSI/pass和缺类信息；`native_geometry.csv`、`native_by_class.csv`：尺寸、160损失及端点估计风险；`overlap_summary.csv`：与两套24港来源的实际交集。

后续方法复验可用OSS2源合法R50 layer3 weighted GAP→Ridge→T，再比较既定additive方法。冻结方法不等于固定24港监督权重跨域直推；不含oldAD的邻域模块，也不应称24港完整系统复现。使用合法inner fit重拟合预处理与读出，T是否二级OOF必须如实说明。评价应保留全部2746条和两条已知图像异常，标明其风险，不通过删异常美化表现；若将来改变队列须另行明示。

本可行性任务已完成。新的reader训练是后续另行授权阶段，输出应另记于readers/，不反写本报告为已有分类结果。
