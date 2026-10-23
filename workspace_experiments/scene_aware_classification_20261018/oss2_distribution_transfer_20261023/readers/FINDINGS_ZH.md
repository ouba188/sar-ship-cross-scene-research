# OSS2源合法Ridge读出完成

2026-10-23。2746条、2048维layer3特征；全部6个outer与24个inner reader已拟合并保存，耗时10.94秒。仅CPU运行，未训练校准T或增量头，未计算测试准确率，未修改主流程文件或操作Git。

## 拟合与选择

严格读取既有indices.npz，不重新分包。每个fit分别执行StandardScaler、PCA128（randomized，seed17，iterated_power4）、post StandardScaler及加权Ridge。权重按fit内MMSI重复数和各类独立MMSI数计算，使八类总权重相等。输出logit乘3。

六折均仅按outer源validation balancedNLL从1/10/100选择alpha=1；相同风险保留先出现alpha。inner沿用该alpha，独立重拟合全部预处理及Ridge。共30次PCA拟合、42次Ridge拟合，保存30个选定reader。全部alpha候选值见alpha_candidates.csv，逐折选择及耗时见selections.csv。

## 校验

- outer train/validation/test两两MMSI和pass隔离；每个inner fit与held同样隔离，且均为outer train子集。
- 24个inner fit均八类齐；部分held缺少稀有类不被错误视为fit缺类。每折四个held恰好覆盖全部outer train一次。
- 所有30个reader对全部2746行核对磁盘W/intercept与原预处理+Ridge：最大绝对误差7.105427357601002e-15，top1变化0。
- 从磁盘保存的scaler/PCA/post/head参数重建完整投影，最大误差3.752553823233029e-14，top1变化0。
- 相对于统一float64预测，float32预处理预测最大误差3.231221713306809e-6，所有30个reader合计top1变化0。具体值见linear_checks.csv。
- 六个test索引集合恰好覆盖cohort全部行一次；未用测试标签选alpha或评价性能。

## 文件接口

models/fold{f}_outer.npz及fold{f}_inner{k}.npz保存W(8,2048)、intercept(8)、fit、held、fit_weights、head_coef/head_intercept、scale_mean/scale_scale、pca_mean/pca_components、post_mean/post_scale、alpha和logit_scale。outer的held是validation与test索引并集，inner的held是实际OOF待预测行；不把outer未使用的其他purged行列为held。h0/b是线性重建参照，不用于额外训练。

fold{f}_inputs.npz保存sample_ids、train、validation、test、base(2746,8)、oof(2746,8)、inner_bin、selected_alpha、valid、logit_scale。base为outer对全cohort的float64分数；oof仅outer train行有效，其他行是0，inner_bin为-1。后续训练必须用train索引读取oof，不可将其他零行视为OOF预测。

本阶段OOF限定为Ridge拟合及其全部预处理；alpha由合法outer源验证统一选择，不是每个inner再独立选参。后续校准/增量流程是否OOF属于各自协议，不据此宣称整流水线OOF。
