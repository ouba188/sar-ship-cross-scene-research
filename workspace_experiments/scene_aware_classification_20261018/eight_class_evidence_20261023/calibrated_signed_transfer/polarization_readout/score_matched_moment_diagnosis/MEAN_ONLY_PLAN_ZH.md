# 下一最小对照：同辅助W的均值输入替换

日期：2026-10-23。状态：仅设计评估，未实现、未训练。centered修正报告及四图已先完成，原uncentered结果保留。本计划不修改父脚本、模型或缓存，不操作Git。

## 判断与问题

建议采用同11维descriptor、209参数shared head的mean-only替换，暂不采用7维145参数版本。它是当前最直接的最小对照：同图像、同辅助W、同极化/缺失信息下，普通加权GAP均值能否取得局部sigma/kappa读出的效用，而不是继续更换小头。

已直接读取`strong_calibration_increment/pooled.csv`：local对强B的test净例为26/41/11，ΔOA=.5481/.8642/.2319pp、ΔBA=.6266/1.2317/.5459pp；extra_score为11/-8/29。现有结果支持强B之外存在可利用的图像读出路径，但额外score不包含辅助W的图像均值，尚未排除“只需另一辅助分类读出”的解释。centered替换诊断证明的也是相对近似score匹配的本图信息效用，不是局部形状独立必要性。

## 唯一新增臂

拟在`polarization_readout/strong_calibration_increment/mean_only_control/`隔离实现。只新增mean_only，3fold×seed17/29/43，共9条轨迹。冻结既有强B及local/extra_score结果，不重训对照，不新增搜索。

缓存`mean_pol[reader,N,pol,class]`来自与moments相同的原160 crop/pad、valid footprint加权池化、每C4旋转后取均值再C4平均；不是把padding当候选的无权整图GAP。`extract_moments.py`实际保存的是每极化W作用于激活的加权响应均值，不含intercept；两极化求和再加对应reader截距才重建辅助ridge分数。均值可为负，不取绝对值、不裁成sigma、不人为分摊截距。

对每样本、每类c，令Bc为强B逐行类中心化分数，m[p,c]为缓存均值，mc[p,c]=m[p,c]-mean_class(m[p,:])。11维输入固定为：

`[Bc, mVV, mcVV, mVV, mcVV, mVH, mcVH, mVH, mcVH, validVV, validVH]`。

两个原moment槽均替换为mean，列名使用`mean_vv_copy1/2`等，不称为sigma/kappa。缺失极化的m及mc在原始descriptor层为0，valid flag照旧；标准化后缺失槽可非零，这与原实现一致。无需乘法carrier：复用`standard_controls.Head('free_shared')`，它只读descriptor，输出每类shared MLP残差并减类均值，忽略carrier；不得误用226参数tanh×kappa头。

## 冻结数据流与预算

1. 使用mean_pol同一reader_names：validation/test用fold外层W；源训练逐行用四个inner reader各自held行，与现有moments的OOF装配完全一致。复用已通过的mean-logit检查，不提图、不拟合新W；核对行ID/reader对应、valid及重复槽一致即可。
2. 强B严格复用`strong_calibration_increment.calibrated`：源训练为已有冻结T作用于oldA+D OOF，validation/test为同一T作用于outer分数；冻结T不是交叉拟合校准器，不能写成整条流水线OOF。逐例B与已保存local基线重放一致。
3. scaler按本臂源OOF train descriptor在(sample,class)两轴求每槽均值/标准差，floor=1e-8，与原free local相同规则；不沿用sigma/kappa数值尺度，也不在validation/test拟合。两个副本的标准化值应相同。
4. `Linear(11,16)->GELU->Linear(16,1)`共209参数。各seed随机初始化与既有local对应，末层为零，epoch0严格回到同一强B。共享头、centered输出、无独立新增参数。
5. 完全沿用jointCE、全源批次、AdamW lr=.003/wd=.01、200steps；每5步只以source validation J=.5OA+.5BA选点，含epoch0，平局保留最早，不以NLL破平局。沿用父实现训练fp32、评价delta转float64后加原B；最终保存头重放验证选中J和top1，不放宽误差。
6. 9个选点全部冻结后评价test一次。保存预测、source曲线/选点、OA/BA/NLL/Brier、逐类/pass/MMSI统计、训练成本；相对同seed冻结local/extra_score/B配对，pass及MMSI两种区间均报告。固定seed17首ID四结局回看原VV/VH，可复用原renderer，但标注强B，不伪装空间矩贡献图。

## 可回答与不可回答

- mean-only若与local接近或更好：当前局部形状优于均值的证据不足，可以保留辅助图像读出有用这一层，不能据此宣称所有局部信息无用。
- local若在配对源端及test显示可重复优势：支持本冻结预算下局部分布表示较均值替代更有效。逐类/翻转样本与置信区间共同限定范围，不要求处处赢，也不以单个CI跨0一票否决。
- 这两臂不是嵌套信息集合。即使local胜，也不能直接声称给定均值后sigma/kappa的严格条件增量已经证明；若后续需要该更强论断，最小缺口是显式控制均值后的残余效用，但本轮不追加该臂。
- 重复槽匹配参数张量、初始化与预算，不增加独立信息；均值臂输入秩更低，重复输入可改变有效第一层权重组合和优化条件。因此不称“有效容量完全相同”。7维145参数均值头虽更简洁，却同时改变参数量、初始化和优化，不作为本轮优先对照，也不两套都跑。

当前到此停止：仅计划，新增训练0、提取0。待明确启动后执行这9头，不扩展设计。
