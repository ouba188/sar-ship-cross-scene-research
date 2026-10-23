# 固定双向 attention 响应层归因与小干预

仅分析父 `test_review_cases.csv` 已选 seed17 四例：5030 rescued、7310 harmed、3494 CC、3504 WW。使用真实 attention9 中对应 fold 的 checkpoint、tau、源 scaler、合法 outer W；不拟合、不改父、不Git。结局与最终最高非真类均来自 attention 的预测，不继承旧 additive 案例标签。

## 固定路径与闭合

原窗 native160 crop/pad、VV/VH 分别、原 footprint area 权重及四旋转不变。重提中心化类别响应 r，重放父缓存 h/b 和完整分数。每 pol 每 view：`a±=softmax(log(w)±r/tau)`，`q±=sum(a±*r)`，`h=(q+-q-)/2`，`b=(q++q-)/2`，再 C4 平均。仅这些槽随 `r(t)=t*r` 变化，B、mu、valid、tau、scaler、head、真类和最终竞争类均固定。

IG对实际r张量求导，积分64中点；闭合容差 `1e-5+1e-4*abs(M1-M0)`，只有不满足才128，不调整模型。闭合目标是真实同head分差减r=0分差，不是整个attention分支输出；r=0时h/b=0但经scaler和已训练head后残差不保证为0。C4梯度已含1/4，逆旋后求和，不再平均。零pool权重格不显示贡献。

## 固定响应干预

对每例、每pol单独干预，另pol保持真实r。每个C4 view按 `abs(sum_class IG)` 排序，在正权重格中选择 `ceil(0.2*n_valid)` 个高格；同数低格为对照，优先在每个相同pool权重层内选最低绝对IG且不与高格重叠。若层内不足，剩余最低IG格补足，逐view记录数量、实际面积质量与匹配偏差，不改阈值追求结果。四view各自同数，不宣称它们是独立样本或严格相同原像素遮挡。

选中格所有8类r先置0，再在整个有效支持内减去每类加权空间均值以恢复中心化；权重不变。原有跨类中心化因同格共同置零而保持（数值误差另记录）。因此最终选中格未必仍为0，未选格也受重新中心化影响；这是“置零后投影回中心化响应空间”，不是纯单格删除。保留B/mu/valid，此响应干预不保证存在对应真实图像。记录原始/干预分差、实际赢家、有效面积和响应变化范数；不重选竞争类，不调干预强度。

## 解释边界

10x10粗layer3网格、理论RF267大于160画布，不是像素归因或物理散射因果。0基线为此坐标下均匀类别响应的零偏离，不是无船图；径向路径保持中心化，但中间表征未必图像可实现。IG正贡献不保证删除该格降分。四固定例只定位已训练模型内部作用，不估计总体机制比例。

引用 Sundararajan, Taly, Yan (2017), *Axiomatic Attribution for Deep Networks*, PMLR70，第3节式1/Completeness。主流程已核读官方正文及第4节边界，本支线复用，不重复下载：`https://proceedings.mlr.press/v70/sundararajan17a/sundararajan17a.pdf`。
