# 固定模型共同增益诊断

2026-10-23。只写本目录，不训练、不拟合、不改父级或其他支线，不Git。输入为old父own_regions_full和new partition_replication/centered_own_full保存模型，六折各三seed。

有效组由selected_kind的原神经通道结构指定：layer3对应16..31，stem+layer3对应0..31。禁止按逐样本q大小筛组。a=0.5tanh(net(env))，有效组算术均值m，contrast=a-m。

精确分解delta=m*sum(q)+sum((a-m)*q)。固定同一A、输入和模型，重放original/common_only/contrast_only；没有重新选择epoch。保存逐例分数、OA/BA/class/net、样本倍率均值/标准差和分解/原模型回放检查。组内标准差ddof=0。

读取centering_probe/origins与partition_replication/origins的outer W/h0和fit索引，验证原q及A。不重新拟合ridge/PCA。使用与ridge相同的类别/MMSI训练weights计算原特征加权中心h0w，直接仿射评价A(h0w)，与3倍加权onehot均值比较。全部三类都有样本时该均值应为[1,1,1]。

原中心b=A(h0)，共同增益为(1+m)*A(h)-m*b，并非一般的纯temperature。另以固定相同m代入neutral-center q，验证正倍率下argmax不变；此项只是数学/数值检查，不假称训练过neutral模型。共同增益仍可改变概率与NLL。

条件区间复用新划分分析的固定预测bootstrap：三个seed先按目标平均正确性差，2000次、种子20261023，pass/MMSI分别报告，不择优；不覆盖训练和划分不确定性。原图按每队列seed29的original救回/误伤/CC/WW，各取首ID并优先无外部邻域，复用原尺度VV/VH缓存。包括触发诊断的new pair_0033132。不以案例替代总体统计。

center选择属于设计选择，不自动判为bug；本次拆分不是重新训练的容量匹配消融，也不证明所有区域信息无用。下一步依据实测共同项/差异项与neutral恒等式决定，不预设创新。
