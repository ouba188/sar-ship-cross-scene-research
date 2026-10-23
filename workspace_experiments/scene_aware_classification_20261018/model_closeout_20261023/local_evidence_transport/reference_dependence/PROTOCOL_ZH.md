# 固定边际参考联合证据可用性试验

2026-10-24。先写协议，再运行。仅在本目录写入代码、CSV、JSON和本地NPZ；不训练骨干、不做原图面板、不操作Git。主代理负责原图分析和阶段归档。本轮不宣称强创新。

## 输入和角色

只读复用`../query_evidence_pooling/details.npz`中的k/a/pi/valid，以及同目录`local_scores.npz`中的bank/idx。k轴为(query,group,query-slot,class,reference)，a为(query,group,query-slot)。源表和完整模型来自`../../regional_utility_pilot/upstream/`。

固定1741条查询：utility_train 644用于系数训练、尺度拟合；selection 612仅选checkpoint；evaluation 485只汇报。源64条参考不变。冻结`strategies_seed17/29/43.npz`的`logits[:,2]`作为完整base。三个seed对应已有base差异；本轮全批量零初始化优化是确定性的，不把三者当三组独立样本。

## 算子和有效性

6组依次为VV body/side/end、VH body/side/end。

`u[g,c,r] = logsumexp_i(log(a[g,i]) + k[g,i,c,r])`。

单区域`L[g,c]`是该区域所有有效参考的等权logmeanexp(u)。用于头的6个共有输入是每张query、每区域的类别中心化L，再按utility有效项RMS标准化。类别中心化只用有效类；当前有效性应核实而非假定。

预定7对：(0,1)、(0,2)、(3,4)、(3,5)、(0,3)、(1,4)、(2,5)。每对每类取两个区域共同有效的参考集合，重新等权归一；不得分别用不同参考集合计算三项。

`D[g,h,c] = logmeanexp_r(u[g,c,r]+u[h,c,r]) - logmeanexp_r(u[g,c,r]) - logmeanexp_r(u[h,c,r])`。

query两区域必须均有效且至少一个共同参考。一个共同参考时D为0，是合法观测；没有共同参考时该对无效。原始缓存将无效项保存为NaN并另存mask。尺度拟合排除无效项，线性头以mask明确略过其贡献，不把填充0当观测，不增加缺失性特征。

## 六臂

- marginal：仅6个共有单区域输入。
- joint：6个共有输入+7个真实D。
- shuffle101/211/307：同样输入结构，但第二区域的共同有效参考在每个(pair,class)内做固定随机置换；同一映射用于所有query。随机种子只决定这些置换。参考有效性、参考质量、query、支持和单区域L完全不变。
- nonlinear：6个共有输入+同样7对的标准化单区域L乘积。该臂不读取配对后的reference联合关系，使用与joint完全相同的pair有效mask。

逐query验证三组shuffle的两个边际logmeanexp都不变，并检查共同参考集合不变、第二边际响应多重集不变。保存逐query最大误差、各pair/类的映射和有效率。发现不符先修实现，不以容许改变边际追求收益。

joint和三组shuffle的7列共用真实joint在utility有效项拟合的逐列RMS，避免置换同时改变特征缩放。nonlinear的7列使用其自身utility有效项RMS。6个共有列所有臂共享同一RMS。RMS近零列使用1作除数，不放大常数或关闭真实微弱信息；阈值固定1e-8。不用selection/evaluation拟合尺度。

## 训练和选择

每臂每base seed独立训练线性残差头：`delta[c] = sum_f theta[f] * feature[c,f]`，输出再作类别中心化；系数跨类别共享、无任何bias，marginal6参数、其余13参数。最终`score=base+delta`，不额外搜索融合倍率。

theta全0初始化，Adam lr0.01，200步，全量utility；损失为`0.5*普通CE均值 + 0.5*各类CE均值的均值 + 0.01*sum(theta^2)`。明确采用平方和L2，使每个共享系数的正则强度在6/13参数臂一致。不存在额外权重衰减。

checkpoint仅0/20/50/100/200。selection上的完整模型exact_j=(OA+BA)/2用精确分数比较；严格提高才替换，平局保留较早checkpoint，因此0可回退。evaluation不选臂、不选checkpoint、不调超参。

## 输出与验证目的

先验证缓存轴/顺序、单区域L重放、置换边际、单参考D=0和简单正负依赖例子；这些检查分别排除错索引、错误聚合、失效的控制、算子符号错误。再运行6臂×3seed共18组。保存特征/有效性/尺度、全部checkpoint系数、所选预测、训练与选择曲线、三角色OA/BA/NLL/Brier、逐类结果及相对full/marginal的救回误伤。

receipt只有18组完成、所有有限性/0步重放/预测重放检查通过后才标记complete。收益只说明当前固定表征、区域和支持库下联合证据的可用性；不能证明物理机制、跨数据集泛化或强原创性。原图回看由主代理接续，当前支线不重复制作图。
