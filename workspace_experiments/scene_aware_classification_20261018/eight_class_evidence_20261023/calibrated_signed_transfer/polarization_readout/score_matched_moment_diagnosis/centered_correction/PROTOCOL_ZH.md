# Centered B匹配修正：只改距离坐标

2026-10-23。主代理在设计一致性检查中发现实现遗漏原先约定的逐行中心化；这是子实现偏差，不是用户改标准。先读原结果并运行check_centering.py：源B逐行类均值并非恒定，最大范围0.215068；评价集最大范围0.103974；合法候选边距离最大改变0.162248；918条fold/seed/role记录改变donor。因此必须补修正，不能假称原始/中心化等价。

只本子目录写修正输出，原uncentered predictions/统计/报告保留，不用修正结果覆盖。复用父run_diagnosis.py参数化输出位置与centered=True，并复用同一分析器。没有新头、训练、epoch选择、阈值或匹配搜索。

唯一变化：先对源OOF train B逐行减8类均值，再按样本轴求8类各自mean/std；评价B也逐行减类均值，用这些源尺度作8维欧氏距离。匹配分组仍按原B预测类和validVV/VH、同fold/seed/role、不同MMSI；贪心最短边、ID平局规则、两两整组矩交换、未匹配原样完全不变。

头实际输入的B和冻结descriptor scaler不变。original分数和top1仍须重放父free_real缓存；以预检查centered donor清单核对实际修正配对。sourceval/test独立，不混合标签挑donor。匹配尺度与head scaler严格分开。

完整输出同原诊断：覆盖/距离/MMSI/类别/同类率/未匹配原因，B-original-swapped OA/BA/NLL/Brier、逐类及pass/MMSI配对CI。补充逐类CI为original-swapped，source/test与3seed分开，两个块单位均保留，不改干预规则。

固定seed17取original→swapped四结果首ID；4query+实际4donor，左右配对、VV/VH各native及160窗口，短文件名panels/matched_rescued.png、matched_harmed.png、matched_CC.png、matched_WW.png。实际查看修正配对图，ID保留标题/CSV。无对应结果则如实记录。

比较修正与原始版本的实际效用，不按哪版test更好选规则。修正版本为本指令主结果，uncentered为保留的实现偏差历史。近似分数匹配不是条件分布抽样，条件性效用不等于像素因果或优于最佳校准系统。完成这一修正和报告即停，无Git。
