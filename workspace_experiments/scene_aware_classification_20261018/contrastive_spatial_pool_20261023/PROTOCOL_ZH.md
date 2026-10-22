# 最小对比空间池化：运行前固定协议

2026-10-23。仅写本目录；不训练NN、不重拟合PCA/ridge、不改变原划分/A、不改父脚本、不Git。本协议先于提取及测试结果写入。

## 范围与依据

三类5695原尺度VV/VH观测、4312 MMSI、163pass、15port；固定20261023新六包、test k/val k+1及既有MMSI隔离。冻结各fold layer3 A、原source拟合W和alpha=1。此前四例粗空间读出存在局部正负抵消及源参照项影响，但4×4/5×5和大RF不能证明船体/背景分离。本轮检验局部证据分布的最小读出，非几何排列创新，非物理去噪。

这是CSRA-inspired frozen mean-preserving baseline，不声称官方论文复现。与上一轮门控不同，本轮不使用env、d、h0或独立可训练分类头。源参考项不进入局部残差，原A独立保留。

## 三个预定臂

对每个极化和每个整90度旋转分别提取原尺度layer3 map F，Fmean为该map空间均值。W_pol,c为该fold源fit的原通道类方向，已含原读出的因子3。

```text
v[c,p] = sum_channel W_pol[c,channel]*(F[channel,p]-Fmean[channel])
r[c,p] = v[c,p] - mean_class(v[:,p])
class_residual[c] = sum_p softmax_p(v[c,:]/1)[p]*v[c,p]
contrast_residual[c] = sum_p softmax_p(r[c,:]/1)[p]*r[c,p]
contrast_std[c] = sqrt(mean_p (r[c,p]-mean_p(r[c,:]))**2)
delta = mean_C4(delta_VV + delta_VH)
score = frozen_A + lambda*delta
```

std使用总体标准差(ddof=0)，不是样本标准差。T=1，不搜索。先各旋转非线性池化，再C4均值，不先逆旋转map平均。三个残差在每类空间响应恒定时均为0。双中心化后r每类空间均值与每位置类均值均为0；残差非线性池化后不再强制类均值为0。

每fold/每臂lambda仅用既有源validation balanced NLL从[0,.1,.25,.5,1]选择，含0；数值差<=1e-12视作平局，按列表顺序优先更小lambda（含0）。18次确定性选择，不做训练seed重复，不把它们称18次独立训练。test不参与选择。delta计算不读取标签；源val标签仅用于lambda选择，测试标签仅用于最终评价。没有新的训练/OOF需要，不声称整流水线OOF。

## 提取与数值

复用原R50权重和原PNG预处理：灰度/255、复制RGB、ImageNet标准化，模型fp32/eval/frozen，关闭TF32。原尺寸shape分桶，共享同一F计算六fold W，不resize、不插值、不额外输入padding。网络内部原padding保留。map激活为fp32；W/局部读出及池化使用float64保留保存W精度。v用`W@F - mean_p(W@F)`计算，与上式线性等价，不接触h0。

逐图C4 GAP与既有deep_features.npz比较（atol/rtol=5e-5），逐fold由重提GAP和保存W/原截距恢复A，记录精度误差和top1变化；实际预测始终使用原保存A。输入base/索引、source和W均按instance_id对齐。

必须验证：均匀空间权重读出v/r的delta为0；r行/列均值0；加入同一class-common空间场后两个contrast残差不变；共同位置置换后三臂残差均不变；每类整图常量响应后三臂为0。检查真实map及确定性构造输入，不把不变性当分类收益。另对少数map用独立NumPy公式核对GPU结果，保存小验证map，不保存全量F。

## 交付与分析

保存小delta NPZ、18组lambda候选/选择、逐例预测、逐折/pooled OA/BA/NLL/balanced NLL/Brier（每样本三类平方误差之和）、类别、采集/港口、救回误伤、提取/选参/推理成本。按pass及MMSI分别做固定预测配对块区间，二者都报，不择优。

原图至少覆盖四种结局；按每臂ID序首个救回/误伤/CC/WW，无该结局则明确缺失，不造例。同时保留前次既定四例预测变化。VV/VH分开、原窗大小逐例标注，显示实际记录OBB，不命名物理响应，不将代表例当总体比例。std臂用于辨别是否仅局部离散强度即可解释收益；三个结果原样报告，不按测试只保留赢家。

只留小delta与小验证响应map，不存全量深层F。结果用于重复开发数据上的读出假设检验，不是独立新数据确认。
