# 固定候选的旧采集划分复验

2026-10-23，运行前冻结。仍为同一三分类 5695 条队列，不称独立外部数据；使用之前 EXP 六折采集划分，而非新 partition seed20261023。每轮源训练/验证/测试排除相同 MMSI 与采集重叠。

目的：判断 jointCE+J signed-only 的当前正收益是否能在另一个既有划分下保留，并与同目标普通分数头和无符号局部读出比较。不开发新架构，不根据新测试结果改选点、改特征或改超参数。

准备：保留历史 fixed-layer3 A 的精确分数与 W，源 alpha=[1,1,10,1,1,1]；源训练内部按其余四个采集包 OOF，重新拟合相同 View/PCA/Ridge，但采用该外层已选 alpha，与新划分的 OOF 流程一致。内层 held 船只从 fit 中排除。将各 inner/outer W 作用于同一 native VV/VH layer3 C4 图。只检查缓存对应与均值分数重构，避免错误的类轴/样本/划分混用。

冻结四臂：signed-only113参数、score-only115参数、conditional-std81参数、linear-distribution3参数。训练均为 jointCE=.5 ordinaryCE+.5 balancedCE，AdamW .003/.01，200步，每5步源验证；三个非线性臂seed17/29/43，线性确定性一次。共60轨迹，只以J=.5OA+.5BA为主选择（含epoch0、平局取最早）；源balancedNLL选点只保存次要结果，不按测试替换主候选。

报告基础/修正 OA、BA、净救回、类别/采集分布、配对不确定性与三种子波动。固定seed17各臂按ID选救回/误伤/持续正确/持续错误原图，VV/VH分开回看；不把热图称为物理因果，也不要求全部类别或全部采集获胜。无原8类A+D更改，无5%交接。
