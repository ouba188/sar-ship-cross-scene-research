# 冻结执行协议

2026-10-23，任何新提取/拟合前落盘。沿用PLAN唯一two_sided_attention，不增加结构、温度或参数搜索。

本轮由当前主流程按用户持续推进授权实施源9头，并已一并授权其后固定历史test推理；不是看源或test结果后再伪称用户另行授权。

1. 源阶段只提原3fold合法inner-held train/outer validation角色，8750对象并集；同PNG/160 crop_pad/coverage/valid、ImageNet R50 layer3、C4各自计算后平均，源OOF W及outer W不拟合。fp32 encoder、float64响应/聚合。每批最多16对象。
2. 每fold每pol的tau仅为源OOF train有效极化全部8类缓存C4平均sigma_e中位数；固定后不得修改。alpha_pm=softmax(log(w) +/- r/tau)，q_pm=sum(alpha_pm*r)，h=(q_plus-q_minus)/2，b=(q_plus+q_minus)/2；双向同一绝对温度。w=0严格不参与。
3. 用h/b替换sigma_e/kappa_e及其类中心化槽；共同mean/B/valid不变。源训练tokens重新fit descriptor scaler，同初始化、additive266参数、零末层，jointCE、AdamW .003/.01、200steps、seed17/29/43。每5步源J选点incl0、最早平局，共9头。full旧9头只重放。
4. 保存全部源曲线/选点/预测/配对/原图、源报告，并记录selection_frozen及source_report_frozen时间。**之后**才读取test参考分数并补提尚未计算的test outer合法角色。test共4744，全部固定模型推理，不新增fit或调整温度/选点。
5. 以full和B为参照报告源/目标OA/BA/NLL/Brier、类别/采集/有效pol、救回误伤。test固定预测pass及MMSI块区间均保留；非独立新队列确认，T不是二级OOF。原图seed17每集合按ID/fold固定首个rescued/harmed/CC/WW，缺结局记录缺失，不换定义。
6. 校验cache均值/矩/GAP、合法OOF映射、概率/遮罩/反向与置换、初始化及选点重放；保留实际误差、wall time与内存。报告正负结果后停止，不Git，不写本目录以外。

实施代码不得执行父main或修改父global路径。模型权重/NPZ仅本地。历史test只作为已开发队列上的合理替换对照，不称官方CSRA复现或首次attention。
