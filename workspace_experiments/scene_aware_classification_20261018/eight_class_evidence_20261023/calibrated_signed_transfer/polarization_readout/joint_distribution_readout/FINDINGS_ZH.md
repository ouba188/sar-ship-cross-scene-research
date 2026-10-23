# Joint / Additive 18头分析与固定分支诊断

## 结论

已有18头保持冻结。本次只新增分析、配对CI、8张原VV/VH回看及两种固定分支删除重放，没有优化或重新训练。

**additive 测试 OA/BA 三个 seed 都高于 joint，当前不支持“均值和分布的学习性交互是必要的”。** additive 相对强 B 三 seed OA/BA/NLL/Brier 都改善，是可保留的同范围正结果；但相对独立 mean-only，OA和NLL/Brier较差、BA较高，不是全面胜出，也不是已经解决全部类别取舍。

固定新additive内部的诊断确认：local支对均值支的原油/offshore错误确有部分保护和新增救回，并非沿用旧独立local的故事；同时它损伤chemical，整体OA贡献小且不全正。mean支承担较多OA收益，local支提高BA；“一支足以解释全部效益”和“交互必要”都不符合现有结果。两支还都输入B和valid，因此不能把local支全部效应自动归因于sigma/kappa的新图像信息。

## 模型与评价边界

- joint：同Bcenter + 每pol(mean/center_mean/sigma/center_sigma/kappa/center_kappa) + valid，15→16 GELU→1，273参数。
- additive：mean分支7→12→1，109参数；local分支11→12→1，157参数；合计266。输出相加后去类公共项。两支共同训练，**不是把旧独立mean/local头直接相加**，宽度也不同，不把删除后的均值支当成旧mean-only模型。
- 延用3fold、seed17/29/43、200步、jointCE、sourceJ含0/每5步/平局最早；同强B=T(oldAD)、同源OOF均值和矩、同源train标准化。T非二级OOF，保持两阶段共享源监督边界，不声称全栈OOF或样本外错误监督。
- 全部选点在test标签评价前保存。源validation仍参与选点，不是独立确认；915折暴露/816唯一对象，测试4744唯一对象/2973MMSI/85pass/18港。没有把seed或重复validation对象当独立样本。
- 下述“local支”专指本次additive里共同训练的157参数支；“独立local”专指既有209参数基线。

## 全部源/测试结果

OA/BA为%，差为百分点；净数相对强B。完整基线指标在pooled.csv，所有与B/mean_only/local/extra_score及两臂间的配对结果在paired_metrics.csv。

| 集合 | 臂 | seed | 净数 | OA | BA | ΔOA | ΔBA | NLL | Brier |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
|test|additive|17|+46|49.3255|42.5538|+.9696|+1.0087|1.420296|.664118|
|test|additive|29|+45|50.0843|42.5095|+.9486|+.4129|1.401323|.652839|
|test|additive|43|+75|49.1990|41.7748|+1.5809|+1.1271|1.449404|.668434|
|test|joint|17|+40|49.1990|42.1923|+.8432|+.6472|1.419740|.663836|
|test|joint|29|+37|49.9157|42.2944|+.7799|+.1978|1.402707|.653144|
|test|joint|43|+37|48.3980|40.9451|+.7799|+.2973|1.454910|.671451|
|sourceval|additive|17|+13|56.2842|48.4225|+1.4208|+1.5923|1.341216|.617544|
|sourceval|additive|29|+7|57.9235|48.9068|+.7650|+1.0433|1.328891|.606332|
|sourceval|additive|43|+14|55.5191|47.2134|+1.5301|+1.2018|1.357498|.622502|
|sourceval|joint|17|+17|56.7213|48.4983|+1.8579|+1.6680|1.337735|.616229|
|sourceval|joint|29|+8|58.0328|48.8373|+.8743|+.9739|1.330983|.607184|
|sourceval|joint|43|+10|55.0820|47.5792|+1.0929|+1.5676|1.359738|.621959|

测试additive−joint的净数+6/+8/+38，OA+.1265/+.1686/+.8010、BA+.3615/+.2151/+.8298。seed17 NLL/Brier仍比joint略差.000556/.000281，另外两seed更好。

测试additive相对既有读出的差值（seed17/29/43）：

| 参照 | ΔOA | ΔBA | ΔNLL | ΔBrier |
|---|---|---|---|---|
|mean_only|−.6535/−.1054/−.1897|+1.2074/+.7312/+.7237|+.015590/+.008951/+.012575|+.008049/+.004108/+.004949|
|local|+.4216/+.0843/+1.3491|+.3821/−.8188/+.5812|−.056087/−.012247/−.030607|−.019488/−.005781/−.011908|
|extra_score|+.7378/+1.1172/+.9696|+1.2932/+1.6470/+.9357|−.038064/−.045162/−.054174|−.013263/−.016617/−.012422|

因此仍有两个不能抹去的边界：独立mean-only OA三seed都更高；独立local在seed29的BA高于additive。不能以新模型名字代替实际取舍。

## 选点与源/测试差异

选中epoch，按fold0/1/2：joint seed17=60/155/5，29=40/85/5，43=110/200/15；additive seed17=195/100/0，29=35/25/10，43=115/0/185。2个additive选择0，真实fallback保留，不剔除。selection_curves.csv保留全部738点。

源pooled additive−joint OA为−.4372/−.1093/+.4372、BA为−.0757/+.0695/−.3658，未出现测试上的三seed双指标同时优势。测试描述不能倒过来当作源选择规则，也不能据此重选模型或改力度。来源内已存在对强B的正路径，但两臂的泛化排序不同；现有数据不足以单独归因优化、类别组成或某种物理域偏移。

## 配对CI

固定预测、2000次paired pass/MMSI bootstrap、seed20261023，按模型seed分别计算；不覆盖重训/选点不确定性、未作多重校正。全部54条区间记录在paired_intervals.csv。下为pass95%CI重点比较。

| 比较 |seed|ΔOA [CI]|ΔBA [CI]|
|---|---:|---|---|
|additive−B|17|.970 [.316,1.609]|1.009 [.178,1.845]|
|additive−B|29|.949 [.206,1.681]|.413 [−.395,1.139]|
|additive−B|43|1.581 [.877,2.298]|1.127 [.488,1.702]|
|joint−B|17|.843 [.233,1.541]|.647 [−.058,1.361]|
|joint−B|29|.780 [−.048,1.641]|.198 [−.557,.972]|
|joint−B|43|.780 [.047,1.556]|.297 [−.496,1.143]|
|additive−joint|17|.126 [−.536,.742]|.362 [−.356,1.037]|
|additive−joint|29|.169 [−.362,.647]|.215 [−.270,.675]|
|additive−joint|43|.801 [.153,1.423]|.830 [.048,1.453]|

additive−mean_only的BA CI依次[.267,2.123]、[−.017,1.364]、[.106,1.286]，OA三个区间均跨零。additive−local seed29 BA区间[−1.626,.064]，seed43 OA/BA区间均正。不能把全正点估计说成全部比较显著，也不能因部分区间跨零否定已观察收益。

## 逐类、港口和有效VH

测试相对B净正确数（seed17/29/43）：

| 类别 | n | additive | joint |
|---|---:|---|---|
|bulk|1290|+20/+19/+31|+24/+26/+24|
|fishing|145|−1/−3/+2|0/−3/0|
|general|439|+14/+2/+7|+9/+5/+12|
|chemical|1480|−11/+12/+24|−4/+3/+3|
|container|519|+9/−3/+1|+2/+1/−8|
|crude|405|+21/+30/+14|+18/+18/+23|
|tug|202|+1/+3/−1|+1/+5/−1|
|offshore|264|−7/−15/−3|−10/−18/−16|

additive并未完全修复offshore相对B的损失；“比均值更保护”与“整个模型在该类超过B”是不同结论。sourceval additive crude+6/+6/+4、offshore+1/−3/−1、chemical+1/0/+3；测试chemical与offshore取舍不可简单归为一致源规律。

additive主要测试港口净数：Port Klang+7/+23/+27，Santos+14/+4/+5，Singapore+14/+15/+9；Jebel Ali+8/−4/+12、Richards Bay−4/+6/+9不稳定。joint对应Port Klang+12/+32/+1、Santos+11/+6/+9、Singapore+3/+12/+7。18港及fold/pass/product完整表已保存，不从聚合港口数推导物理机制。

测试有效VH4436例：additive净+46/+43/+74，joint+44/+37/+37；缺VH308例：additive0/+2/+1，joint−4/0/0。sourceval缺VH82次：additive−1/−1/−1，joint−1/−1/0；有效833次分别+14/+8/+15、+18/+9/+10。各自OA/BA/NLL/Brier见vh_metrics.csv，不能把有效组收益叫VH因果效应。

## 固定additive分支：精确核算与两种删除

对冻结head，在同一实际val/test batch计算原始mean/local net输出，分别去classmean：dm=center(f_mean)，dl=center(f_local)。数学上delta=dm+dl。原forward使用float32先相加再中心化，分析将各原输出转float64独立中心化；与原delta最大数值差3.1828e−7，和保存score重放最大差3.55e−15，全部top1一致。branch_predictions.csv逐类保存dm、dl和roundoff=原delta−dm−dl，图/表的分差可连同roundoff精确闭合，没有把舍入项偷偷塞进某支。两支自身class和误差为0。

仅做 B+dm（删除local）与B+dl（删除mean）两种固定重放，不改参数/scaler/选点/尺度，不做优化。这是已共同训练模型的输出删除，**不等于重新训练单支消融，也不是独立因果效应**。

下表净数、救回、误伤均是“删除后相对完整additive”，不是相对B；net_vs_B另列，避免方向混淆。

| 集合 |seed|删除支|OA|BA|净数vs完整|删除救回|删除误伤|净数vsB|
|---|---:|---|---:|---:|---:|---:|---:|---:|
|test|17|local|49.3887|41.3811|+3|79|76|+49|
|test|29|local|50.0422|41.9237|−2|61|63|+43|
|test|43|local|49.0093|41.0158|−9|57|66|+66|
|test|17|mean|48.2504|43.0290|−51|100|151|−5|
|test|29|mean|49.3044|42.5965|−37|76|113|+8|
|test|43|mean|47.7867|41.4189|−67|68|135|+8|
|sourceval|17|local|55.8470|46.4430|−4|14|18|+9|
|sourceval|29|local|57.8142|48.2295|−1|11|12|+6|
|sourceval|43|local|55.7377|46.8332|+2|13|11|+16|
|sourceval|17|mean|53.0055|46.5264|−30|12|42|−17|
|sourceval|29|mean|56.6120|48.7096|−12|12|24|−5|
|sourceval|43|mean|53.1148|45.9651|−22|8|30|−8|

完整additive相对只留mean支，测试BA增加1.1727/.5858/.7591pp，OA净−3/+2/+9；sourceval BA增加1.9795/.6773/.3802pp，OA净+4/+1/−2。因此local不是完全冗余，但不能说所有OA收益需要它。删除mean对OA损失明显，BA却在test前两seed更高，说明仍有类间取舍。没有因删除结果好看而更换部署模型。

### local到底保护哪些对象

在本次additive中将local加回“只留mean支”的固定预测，逐类净数（即上表删除local的相反数）：

| 类别 | sourceval | test |
|---|---|---|
|bulk|+3/+4/+4|−5/+12/+14|
|fishing|0/0/0|+1/+1/+2|
|general|0/+1/0|+4/+4/+1|
|chemical|−10/−7/−8|−45/−28/−28|
|container|+1/0/−1|+7/−6/+6|
|crude|+4/+3/+1|+24/+16/+9|
|tug|+1/0/0|−2/+1/0|
|offshore|+5/0/+2|+13/+2/+5|

这里确实能在新模型本身观察到对mean支错误的部分修复，而非旧独立头结果迁移。进一步分开“保护B原正确”和“救回B原错”：

| 测试类别 | B正确、mean支错、完整对 | B错、mean支错、完整对 | local新增损失合计 | local净数 |
|---|---|---|---|---|
|crude|11/1/2|13/18/10|0/3/3|+24/+16/+9|
|offshore|10/2/4|4/1/2|1/1/1|+13/+2/+5|
|chemical|1/1/0|2/5/4|48/34/32|−45/−28/−28|

sourceval crude保护1/0/0、救回3/3/1，无新增损失；offshore保护4/0/2、救回1/0/1、损失0/0/1；chemical修复1/1/2、损失11/8/10。这表明offshore的作用较多是保护，而crude也有实质新增救回；chemical代价源端已有。全部两支删除的逐类净数/救回/误伤在branch_deletion_classes.csv；对应保护、B错救回、B对损失、丢失保留支原救回的四类分解在branch_roles.csv。

这不是纯局部矩的因果证明：local支同时读取B，且两支共同优化，删除会改变训练时的组合。只能说在当前固定已训练additive中，这部分数值输出确实承担了上述决策作用。

## 8张实际原图与分差闭合

seed17，test每臂rescued/harmed/CC/WW首instance_id；8面板对应5个唯一对象，全部实际view_image。短名panels/{joint|additive}_{outcome}.png；完整原路径、窗口、epoch、原/最终rival在review_cases.csv。不是8独立图像证据，也不依据图像调参。

| 臂/结局 | ID/原图可见事实 | 实际分数变化 |
|---|---|---|
|additive救回|pair_0004919，general/Busan，VV/VH64均有斜向短亮体、背景较暗，VH亮形与VV不完全相同|chemical→general；固定最终竞争3的margin：−.13193 + mean .12557 + local .20066 = .19431（舍入项−2.6e−8）。mean部分改善但单看此3−2竞争不足翻转，local提供进一步正分差。 |
|additive误伤|pair_0007310，bulk/Fujairah，VV/VH72长斜向连续亮体、周边有纹理|bulk→crude；+.02655 − mean .11074 − local .27146 = −.35564。此例两支同向误伤，不能称local总在保护mean。 |
|additive CC|pair_0003494，bulk/Busan，VV/VH64横向亮体，VV周边纹理更明显|bulk仍对，旧rival2变最终rival1。**固定最终rival1**的B分差3.22562 + mean .44976 − local .67286 = 3.00252。旧最高rival分差2.35965不能直接代入该闭合。local此处降低对最终竞争者的分差但未改正确结局。 |
|additive WW|pair_0003504，bulk/Busan，VV/VH64斜向长亮体|general仍胜；−.64017 − mean .19493 + local .11474 = −.72036。local抵消部分均值负作用但不足救回，总分差仍较B差。 |
|joint救回|pair_0005030，general/Busan，VV64强亮斜体伴细直亮线，VH64框内亮团更分离|bulk→general；margin−.15439→+.45185。joint可救回，但同对象旧mean/local也能修，不是交互独有案例。 |
|joint误伤|pair_0007310，同上原图|bulk→crude；+.02655→−.22945，真类分数增加但crude更多；两臂均有该竞争错误。 |
|joint CC|pair_0003494，同上原图|bulk仍对，rival2不变，margin2.35965→3.07544。 |
|joint WW|pair_0003504，同上原图|general仍胜，margin−.64017→−.71947；强亮外观并未保证类别可分。 |

原VV/VH与实际160crop/pad并排，未配准逐像素相乘；绿色为记录OBB。图题明确joint/additive，additive下图为两支去classmean后的**网络输出**，不冒称物理κ贡献。卷积RF可混合框外/padding，亮线和纹理不被强行命名为物理成因。图像案例只解释本例决策位置，不代表全队列比例。

## 成本与交付

原训练summary记录18头3600步，训练/源选择/保存/重放24.21秒，缓存流程总29.79秒，2个0步fallback，无提图；本次分析新增训练次数为0。缓存训练运行时间不当新增部署推理成本。

主代理随后已用benchmark_readout.py对固定32个原PNG样本（双极化有效、按ID而非预测结果选取）实测辅助图像路径，详见[COST_ZH.md](COST_ZH.md)与pixel_costs_summary.json。在RTX5050Laptop上，从native160输入到R50 layer3/C4、辅助W投影、mean/sigma/kappa及additive读出，batch1/16/32的中位数分别为22.136/7.373/7.762 ms/张；含PNG读取、裁补与mask构造的热I/O路径分别为23.508/8.272/8.698 ms/张。每组3次预热、10次同步计时，B仍从缓存提供；**该阶段读出与warm-I/O成本已测，但整旧AD/T全系统端到端成本未测**，模型/元数据初始化、冷I/O不在上述范围。均值编码参照不含mean-only头，只是低界；batch32分段计时波动不能解释成多计算必然更快。实际代价主要是辅助图像编码，不以小头参数少代替成本评价，也不提前假设与旧主模型共享native前向。像素复算均值/矩/分数最大差2.37e−6/4.99e−6/4.59e−6，32个top1一致；测量口径以成本报告为准，原成本文件未改。

保留原run_joint.py、PROTOCOL、checkpoint、选点和预测不变。新增analyze_joint.py、paired_metrics/paired_intervals/paired_classes、class/port/fold/VH/采集表、源验证去重表、branch_predictions/branch_checks/branch_deletion_metrics/branch_deletion_classes/branch_roles、8面板及review_cases.csv。本次汇总曾因重复n字段拼接失败，已局部修正分析脚本后成功完成，未改训练数据或模型。无新训练/新尺度/新路由，无Git；到此收口。
