# 空间连续性source pilot：九头完成

2026-10-23。**local没有提供超出同容量marginal的selected分类增量；complete在seed17/43改善BA，但它只添加已共同提供的Kish有效数的确定性非线性函数，不携带真实排列信息。** 当前不支持继续扩大此拓扑读出，也不据此否定所有空间信息。固定source pilot已经结束，无test评价或新增网络。

## 已执行范围

source train1542/validation279，共1821图，固定fold0、三seed17/29/43。一次源图GPU提取，仅使用各样本合法inner held或outer W；224保全PNG、不resize/crop，原C4、VV/VH、footprint和冻结B不变。3臂×3seed共9头，253参数，共1800步；总执行38.19秒，包含提取、训练和主案例出图。事后补充单点复算/complete原图不训练、不再次提图。

共同输入为原joint.inputs15加VV/VH Kish有效数2槽，再加VV/VH R2槽，共19；marginal新槽0，complete为完整图R，local为四邻接R。相同seed三臂初始化逐元素相同，共同前17槽逐元素相同。scaler仅source train拟合；200步jointCE、原AdamW预算、每5步source J含0、平局最早。随机图文件仅未训练的设计记录，未混入最终结果。

## 数据与数学闭合

所有1821对象双极化都有至少4个有效格/4条四邻接边；Kish中位train16.99、validation21.08。但211个train和27个validation的Kish不足4，且边相互共享端点、大RF重叠，不能把这些格/边当独立观测。fractional格权重质量占比中位train39.08%、validation37.57%；没有二值化footprint或剔除小支持对象。

E=sum_{i<j in graph} w_i*w_j*(r_i-r_j)^2 /(2*sum_{i<j in graph} w_i*w_j)。每pol每C4先求R=E/σ²，再平均四视图。完整图E_all=σ²/(1-sum p_i²)，所以非零方差时R_all=1/(1-1/n_eff)。邻接图与完整图边数不同，不称纯shuffle，也不声称局部R已经自动去除尺寸关联。

- 重提mu、sigma、kappa对source_moments缓存最大误差均为0，沿用原缓存值作为共同输入。
- 1821×2×4×8共116544个class/pol/view中，实际零方差为0；无邻边样本0。代码对精确零方差显式置0并保存flag，没有加epsilon制造比值信号。
- train/validation各一个保存点、每pol/C4共16个检查：显式所有无向边完整图能量与解析式最大误差2.00e-15；保存r/weight/R后独立CPU枚举边重放比值最大误差1.78e-15。
- 等权2×2 stripes与checkerboard的mean=0、variance=1、kappa=0、E_all=4/3相同，E_local为1和2。这只证明新统计数学上可辨识排列，不证明真实分类有效。
- 9个selected checkpoint均重放相同源验证J。原图标签仅训练/选点及事后展示使用，不参与邻边、footprint或Kish定义。

## 源验证结果

OA/BA为百分数，net相对各seed自己的冻结B。

|seed|arm|selected step|net|OA|BA|NLL|Brier|
|---|---|---:|---:|---:|---:|---:|---:|
|17|marginal|0|0|53.7634|40.0211|1.450252|0.659189|
|17|complete|195|+1|54.1219|44.5762|1.403422|0.638466|
|17|local|0|0|53.7634|40.0211|1.450252|0.659189|
|29|marginal|5|+1|53.0466|42.2596|1.558165|0.698808|
|29|complete|5|+1|53.0466|42.2596|1.557774|0.698686|
|29|local|5|+1|53.0466|42.2596|1.557998|0.698755|
|43|marginal|0|0|54.4803|40.8304|1.454385|0.661718|
|43|complete|140|-2|53.7634|43.4019|1.394167|0.635504|
|43|local|0|0|54.4803|40.8304|1.454385|0.661718|

local与marginal三seed各279条selected top1完全相同，不只是OA碰巧相同。seed29概率输出有微小差异，不能说两模型分数相同。complete与local的top1不同数为49/0/46。

complete相对B的ΔOA/ΔBA(pp)：seed17 +0.3584/+4.5551，seed29 +0.3584/+0.6836，seed43 -0.7168/+2.5715。完整图比例本身只是一种有效数特征变换；共同输入虽已有Kish，有限宽度/预算下预先给出该非线性函数仍会改变拟合。不能把它叫新增空间信息，也不能因共享Kish就断言对照必然完全等价。

## 类别与拟合边界

complete seed17各类net为[-4,0,+3,+1,-1,0,0,+2]，seed43为[0,-1,+1,-5,+1,0,0,+2]。最后一类validation仅6条，多对2例单独贡献BA+4.1667pp；它和其他类别损失共同形成最终BA。seed43正确总数减少2仍提高BA是明确类别取舍，不是全面更强。seed29三臂共同类别net为[0,+1,0,0,+1,-1,0,0]。

selected complete在17/43的普通NLL/Brier改善，但balanced NLL由1.702642→1.831671、1.769568→1.860852变差；J选点与概率风险不是同一目标。该源验证已经参与选择，3seed复用相同279对象，不是3份独立确认。

没有因负结果忽略训练过程：local末步训练jointCE为1.204653/1.293369/1.199644，低于对应B初始训练损失约1.394219/1.501056/1.385816。末步源验证J却为.445892/.460671/.439023，均低于0步.468923/.471321/.476553。它确有拟合变化，但没有稳定兑现验证分类效用；选回0/5/0不能说成接口未接通。训练loss为第200次更新前记录，验证为更新后，完整41点曲线和terminal_summary均保留。不追加训练定位唯一原因。

## 原图实际查看

已实际查看4张双极化面板（3个唯一对象）。主local seed17为epoch0，没有救回/误伤，cases.csv对此明确标available=False；补充两张complete对照四结局覆盖，绝不冒称local救回。

1. **local CC 000557，69×69，bulk_carrier，B/local=0**。VV有中心紧凑亮区及向下延展的较弱结构，VH纵向亮区更宽。有效36格、local60边、complete630边；rotation0真类证据有正负相邻格。selected为0步，图中可见连续性没有作用于本次输出，不能拿此图证明local正确原因。
2. **local WW 000556，41×41，bulk_carrier，B/local=2**。VV上方亮块及下方较弱亮区，VH纵向延展更明显。有效16格、local24边、complete120边；r图有紧邻正负值，但仍无selected修正。原像素与RF粗格不是同一空间尺度，不能从该图命名物理失效。
3. **complete救回000571，29×29，真类2，B=7→complete=2**。VV/VH可见斜向数个相连亮块；仅4格4条local边、6条complete边，Kish约3.99。真类local R为VV1.281/VH1.136，complete两路约1.334。实际选中的是不含排列信息的complete臂，不能将救回归于斜向连续性。
4. **complete误伤000557，B=0→complete=4**。与第一例同一张原图，local仍原样正确；真类local R约.886/.845，complete约1.050。说明同样有可见结构和不同R，并不保证附加统计帮助该对象；不把complete误伤当作local误伤。

`case_CC.png`/`case_WW.png`画实际rotation0 r和local边；`complete_rescued.png`/`complete_harmed.png`画原图与两种实际几何图及C4 R，明确不是证据热图。complete图显示全部无向边，节点大小随footprint权重；图上边线仅示连接，计算始终用端点乘权。图不暗示船体分割、精细定位或因果遮挡。

## 交付与停止

协议/准备：PROTOCOL_DRAFT_ZH.md、PROTOCOL_ZH.md、coverage.csv、coverage_summary.csv。实际实验：run_topology.py、topology_features.npz、extraction_checks.json、math_example.csv、onepoint_replay.npz/onepoint_checks.csv、heads/、predictions.csv、metrics.csv、paired_metrics.csv、classes.csv、selections.csv、selection_curves.csv、terminal_summary.csv、selected_agreement.csv、cases.csv/complete_cases.csv与4张图。NPZ/PT仅本地；没有保存全量R50通道图。

该统计属于已有Geary/Dirichlet类量，不宣称原创。真实可保留的结果是：几何支持及数学可辨识性足以执行一个source检验，但当前固定读出未获得local相对幅度/均值/有效数对照的新增分类收益。数据支持小、RF混合及目标/选点取舍仍是边界，不假定所有排列无用。无进一步网络/画布/partialconv、无test扩围。主流程独立处理calibrated TIFF与PNG变换，本支线未重复探索该数据路径；未改父文件或Git。
