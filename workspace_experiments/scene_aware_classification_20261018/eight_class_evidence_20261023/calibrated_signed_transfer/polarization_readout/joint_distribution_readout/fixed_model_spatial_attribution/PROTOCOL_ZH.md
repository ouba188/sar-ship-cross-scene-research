# 固定266头的空间证据归因

2026-10-23。仅本子目录写入，不训练、不重选模型、不Git、不修改原图/标签/父文件。固定seed17 test四例：pair_0004919救回、pair_0007310误伤、pair_0003494持续对、pair_0003504持续错。读取现有full的最终最高非真竞争类，并在整条路径保持不变。

## 重放与路径

复用原native160 crop/pad、原始PNG、valid、面积footprint、冻结R50 layer3、outer辅助W及C4。重新提取仅四例有效VV/VH，按原weighted_moments重建每pol/view的局部中心化r和归一化weight，复核缓存mean/moments及所选266头scores。浮点重放容许GPU批次带来的舍入量级差，不改模型或缓存来强行对齐。

归因输入是已中心化的r节点，不是原始SAR像元。路径r(t)=t*r，t从0到1；B、缓存mu、valid、scaler及已选head恒定。每pol每view单独计算sigma/kappa，再按原C4均值进入run_joint的15槽，只分布原值及其跨类中心化槽随路径变化。t=0表示原始sigma/kappa为0，标准化后通常不是0，不采用component_isolation的标准化后遮槽。

完整分数为B+center(f_mean+f_distribution)。定义固定竞争类分差M(t)。IG_i=r_i*integral_0^1 dM(tr)/d(tr_i) dt，目标闭合sum_i IG_i=M(1)-M(0)。这只是相对同head零分布矩的变化，**不是整个distribution branch的margin贡献**；f_distribution(0)仍可依赖B、valid、偏置及scaler。独立记录B、mean、distribution0、distributionreal及浮点闭合残差。

## 数值与坐标

- 实际Head保持float32及原参数；矩与r使用float64，按原inputs在进入head时转float32，不另拟合或改变网络精度。
- 用64个中点积分，不采t=0。零方差处使用分段安全sqrt，零输出/零导数，避免先sqrt(0)再mask导致NaN；不通过增大epsilon修改非零矩定义。
- 完整性判据预先固定为abs(error)<=1e-5+1e-4*abs(M(1)-M(0))；64点不满足才试128并保留64记录，仍不满足则如实报告不合格，不无限加点。
- 对实际r张量的全部8类求梯度，保留center(sigma/kappa)的跨类路径；每类、每pol、每view贡献可追溯，不只画真类和竞争类两个r通道。
- 每个view的IG已含C4均值的1/4。各图逆旋回原方向后求和，不再除4；pol求和、class求和及旋回前后总和均应一致。
- weight为0的格不参与统计，其归因应为0，显示时遮罩。以面积pool的粗网格边界映回native坐标，无平滑插值、无padding虚构证据；格子着色是粗读出支持位置，不是其卷积感受野仅覆盖该格。

## 交付边界

4幅实际VV/VH原图、native坐标空间图、分数/分差分解与闭合表，以及本地r/weight/IG缓存。实际查看全部面板。报告layer3网格、有效支持格数及由当前模型卷积配置计算的理论感受野；不得把红蓝命名物理散射、像元因果，或把正IG理解为单格删除必然掉分。若遇epoch0，真实归因为0并保留，不换例。此路径是模型内部表征干预，并非存在对应原图或真实SAR干预。
