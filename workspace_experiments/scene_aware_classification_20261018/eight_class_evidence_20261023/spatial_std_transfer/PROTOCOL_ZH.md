# 八类weighted spatial std固定外推：运行前协议

2026-10-23。只A与contrast_std，不重复三类class/contrast softmax臂；不训练NN、不拟合读出、不扩大参数、不改父目录、不Git。三类结果已完成，本轮是范围外推，不是八类full A+D。

## 冻结数据与输入

读取父`cohort.csv`15251观测、`features.npz`、`coverage.csv`及原view_snapshot，保留既有obs31237排除和跨极化alignment禁用；按保存baseline三个fold的train/validation/test。全15251共享一次F提取并同时用3fold W；只对原4744个唯一test行汇总分类，不能将全量当测试。

必须复用`prepare_features.py`实际路径：`crop_pad(raw,masks_for(view),160)`，不resize；support为裁补后的三个互斥mask之和，padding=0。每C4旋转同时旋转图像和support，`area`将support降采样至layer3的10×10，保留边界cell部分面积权重。图像是原像素裁切/补零，只有mask按原实现面积降采样。若validpol=false，该极化mean与delta均0，不参与候选。

原R50/灰度复制RGB/ImageNet归一化/fp32/eval/TF32关闭，网络内部padding保留。footprint排除局部候选不意味着去除了卷积RF的padding影响。batch按原16目标的顺序组织，便于与既有weighted GAP数值复现。

## 唯一候选

W直接取`baseline/models/fold{k}_outer.npz`，已含原logit_scale=3。每极化、每旋转：

```text
w_p = area_downsample(valid_footprint)_p; Z=sum_p w_p
mu_c = sum_p w_p * (W_c @ F_p) / Z
v_cp = W_c @ F_p - mu_c
r_cp = v_cp - mean_class(v_cp)
delta_c = sqrt(sum_p w_p*(r_cp-weighted_mean_p(r_c))^2 / Z)
delta = mean_C4(delta_VV + delta_VH)
score = original_A + lambda*delta
```

总体weighted std，不做Bessel校正。无h0源参照分摊；A独立保留；没有新门控。class中心化保留三类预定算子，但onehot+intercept Ridge类W之和近零已是结构事实，不宣称其为有效独立模块。空间常量响应std=0，对map及mask联合置换不变，不是几何排列建模。

lambda在每fold源validation balanced NLL上固定选[0,.1,.25,.5,1]，含0，差<=1e-12按从小到大。delta不读标签；标签仅进入源选择和最终评价。若源val全0如实报告，不扩参数、不强选非零、不新增OOF或训练。

## 验证与存储

原批次重提weighted GAP与父缓存比较(atol/rtol=1e-4，沿用父single/batch验证阈值)，独立恢复三fold线性分数，A预测始终用原缓存。记录实际误差、top1变化；严禁把padding当全权重候选或把invalidpol补成有效。

检查加权空间均值0、r每位置类均值0、joint(map,weight)置换不变、添加class-common场不变、常量响应std0、零权重cell赋任意值不影响std。均与独立NumPy weighted std回放比较。

保存`std_features.npz`：instance_ids、valid、mean_pol(3,15251,2,8)、delta_pol(3,15251,2,8)、delta(3,15251,8)。mean_pol是每极化W投影的weighted mean/C4均值，无截距；完整2048维weighted GAP已存在父features.npz，本轮只记录重放差异，不重复大型特征缓存。保存每样本原尺寸/裁补/保留target/rest/footprint coverage。仅少量验证响应map，不存全量F。

输出三个lambda选择、预测/分类/风险/类别/港口采集/块区间、源数据无标签mean/delta统计、成本和原图四结果。原图展示原尺寸与真实160裁补输入及mask覆盖；缺失某结局明确报告。所有结果为固定范围外推，不按测试改算子。
