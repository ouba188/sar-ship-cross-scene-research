# 新采集划分复验

2026-10-23，运行前固定。只写本目录，不改父级、centering_probe 或原数据，不 Git。复用5695目标/4312 MMSI/163过境/15港，三类；不是旧八类任务或独立新增数据确认。

## 划分与读出

将按名称排序的163过境使用 numpy default_rng(20261023) shuffle，再按索引%6分配六包。第k包test，第(k+1)%6包validation；先去掉val中test MMSI，再去掉train中test/val MMSI，完全沿用旧规则。所有目标恰有一次测试。

固定完整冻结ImageNet R50的layer3双极化C4原尺度均值2048维；不比较或按测试选择其他读出。每折源训练拟合standard scaler、128维PCA(randomized,17,iterated_power4)、post scaler。源验证balanced NLL仅选ridge alpha=1/10/100。A保留直接读出的分数。源内四包逐包重新拟合上述全部及ridge和h0；排除held MMSI，alpha沿用外层源验证选择，不再内层调参。

## 四臂

- A：不调节，无随机头，结果只记录一次而非伪造三份基线。
- centered_own_full：q_g=W_g(h_g-h0_g)。固定原layer3通道按VV/VH各8组，共16有效组，保留32槽的前16槽为零。
- uncentered_own_full：qraw_g=W_g h_g，使用每个fit自己的W；只替换调节贡献，始终使用同一个A/base或OOF分数。绝不将sum(qraw)当基线。
- centered_plus_bias：centered加共享隐藏层的32->3零初始化加性类别头，3715参数；其余两臂3616参数。此臂同样是自身chip、全量激活，不是父级external加性头。

ownenv严格沿用父controls构造，但按本轮outer source重拟合：源目标stem标准化/PCA32/post-scale分别投影target_fg/bg，unit归一化后无效区域置零，拼接/√2得64维；旧geometry summary以源训练拟合scaler，再取源有邻域样本的均值作为所有目标相同的15维常量。没有逐目标外部邻居输入，也不以邻域存在决定激活；该常量仅使用源聚合信息，并不声称训练构造从未读取邻域统计。ownenv保持外层源拟合，不额外内层重拟合，故仅A/贡献为OOF，不称整流水线OOF。

MLP79->32->32、ReLU、0.5tanh、末层零初始化；17/29/43三个训练seed；200step AdamW(.003,.01)，类别/MMSI平衡CE，每5步源val类别平均NLL选步，改善阈值1e-6并含0步回退。仅一轮54头，不追加搜索。

## 检查与报告

直接A分数使用原transform精度，各臂共用；分解用float64验证centered+中心分数和raw+原截距分别重构直接float64线性图，同时记录相对原A的精度差。保存30次fit的W/h0/origin/fit/held、OOF覆盖、分解误差、划分选参、预测、逐折/汇总/class、时间及54头PT回放；NPZ/PT仅本地。

预定原图选例：seed29的centered_own_full，对救回、误伤、CC、WW各按instance_id排序取首例，优先无外部邻域样本以对应全量自身输入目标；若该类无候选则放开邻域条件。VV/VH展示缓存原尺度，不插值、不增强。案例不是比例或物理因果证明。三seed共享队列；新划分仍是重复开发拆分。与旧轮比较同时改变读出选择和划分，不能单独归因其中一项。
