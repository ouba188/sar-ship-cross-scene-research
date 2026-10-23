# 实现范围与终态交接

2026-10-24。本支线已结束，未追加超参、复杂coupling训练或新审阅。

## 终态

- `run.py`运行进程已确认退出码0；`receipt.json`为`status=complete`。
- 完成6臂×3冻结base seed，共18组，每组200步、5个预定checkpoint，共90条选择记录。总运行约16.08秒，含约2.11秒特征构造；仅为本地缓存后的成本，不含骨干提取。
- 无新骨干训练、无新图像提取、无图表制作、无Git操作。所有本轮新增文件在本目录。
- 原图分析及Git归档由主代理接续；这里未以数值核查替代图像解释。

## 确切实现

1. 直接复用`query_evidence_pooling/details.npz`的k/a/pi/valid，以及`local_scores.npz`的bank/idx。固定源参考库、6个区域和7个区域对均未改变。
2. 对query位置按a做LSE形成u。单区域L使用该区域所有有效参考；D的三个logmeanexp均使用同一个共同有效参考集合，重新等权归一。
3. shuffle101/211/307仅置换第二区域共同有效参考身份，每个seed/pair/class的映射固定，跨query复用；没有打乱缺失性。
4. 无效D在原始缓存中为NaN，另存pair_valid。尺度排除无效项，线性头按mask省略其贡献；计算张量中的零仅为省略贡献的实现，不是新增零响应观测。
5. 各臂共有6个类中心化L，按utility有效项RMS标准化。joint及三组shuffle共用真实joint的utility逐列RMS；nonlinear是同样7对标准化L的乘积，再用自身utility RMS。没有用evaluation拟合尺度。
6. marginal为6个共享线性系数，其余13个；跨8类共享，无任何bias。残差输出再作类别中心化，加到冻结完整base，不另选融合倍率。
7. 全0初始化，Adam 0.01，损失`0.5*普通CE+0.5*类均衡CE+0.01*sum(theta^2)`。utility644训练；selection612用完整模型exact_j严格提升选0/20/50/100/200，平局保留较早步；evaluation485只汇报。
8. 三个seed改变的是已有冻结base，本轮优化为全批量确定性过程；不是三组独立数据证据。

## 已完成的运行内核实

|核实项|结果|
|---|---:|
|单区域L对原缓存重放最大误差|1.78e-15|
|shuffle第一区域边际逐query最大误差|8.88e-16|
|shuffle第二区域边际逐query最大误差|1.78e-15|
|第二区域响应多重集变化|0|
|所选系数独立NumPy预测重放最大误差|1.23e-6以下|
|0步预测与冻结base|逐元素一致|
|常量及单参考D|均为0|

逐query、逐类、逐pair均在运行中验证；`marginal_verification.csv`保存每个query在各shuffle下的最大误差，另有完整参考映射。

VV body-side/body-end在三个角色中有效率均100%。其余涉及VH的5对，utility为561/644=87.11%，selection为561/612=91.67%，evaluation为416/485=85.77%；各臂共享有效性。各类共同有效参考数保存在`common_reference_counts.csv`。

## 结果摘要

evaluation均值，OA/BA单位为%，三个base seed共享485条观测。

|臂|OA|BA|NLL|Brier|
|---|---:|---:|---:|---:|
|冻结full|58.832|40.122|1.34825|0.59353|
|marginal|58.144|39.492|1.30865|0.58207|
|nonlinear|58.763|40.289|1.30375|0.57714|
|joint|58.351|40.394|1.29426|0.57538|
|shuffle101|57.732|39.082|1.31074|0.58197|
|shuffle211|58.351|41.374|1.31401|0.58169|
|shuffle307|58.144|40.080|1.29909|0.57694|

joint所选步为200/0/200。相对marginal正确数净变化+1/+3/-1；其中seed29是回退base优于被训练过的marginal，不能算该seed学到了有效joint。相对full为+1/0/-8。

结论：真实joint有局部及连续评分改善，但分类上未形成超越普通nonlinear和保持边际shuffle的强独立证据。NLL/Brier较好不等于参考联合身份已被证明关键。保留全部正负结果，不追加调参，不称强创新模块。

## 主代理接续所需文件

- `PROTOCOL_ZH.md`、`run.py`：运行前协议与完整实现。
- `metrics.csv`、`summary.csv`、`selection_curve.csv`、`training_receipts.csv`：角色指标、均值标准差、选择轨迹和终态。
- `paired_results.csv`、`class_results.csv`：对full/marginal/joint的救回误伤及逐类计数。
- `features.npz`：idx/bank、u、L、原始D、nonlinear、有效mask和实际训练输入；`scales.csv`记录utility尺度。
- `predictions_17/29/43.npz`：全源表顺序的冻结full及六臂分数，另存idx/bank，便于直接回看实际query与支持图。
- `weights.npz`、`coefficients.csv`：全部checkpoint系数及最终所选系数。
- `reference_permutations.csv`、`common_reference_counts.csv`、`availability.csv`、`marginal_verification.csv`：控制有效性依据。
- `receipt.json`：完成标记、参数、运行耗时、验证误差及范围限制。

NPZ为本地分析缓存，后续Git同步仍按既有排除规则处理。
