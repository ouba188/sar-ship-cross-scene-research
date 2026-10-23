# 分数匹配整组矩诊断：uncentered历史结果

2026-10-23。**本文件保留未逐行中心化B的原始结果，不作为centered修正结果。** 主代理在设计一致性检查中发现实现遗漏原先约定的逐行中心化；这是子实现偏差，不是用户更改标准。实际核对确认不等价，修正独立保存在centered_correction/；原预测、匹配、统计及本报告不以修正数据覆盖。没有训练或新头。

## 结果与含义

在这次固定配对下，原本图矩相对借用矩，sourceval与test三seed的OA/BA/NLL/Brier均更好，支持**当前冻结free_real对本图矩对应关系具有有范围的条件性预测效用**。但借用矩仍保留明显的B之外OA/概率收益；不能把全部增量都归于本图信息，也不能因此宣称胜过最佳分数校准。

B始终为full_model_integration的旧A+D；9个free_real头、epoch、B、scaler均冻结。sourceval为915折暴露/816唯一对象、629MMSI、32pass；test为4744对象、2973MMSI、85pass、18港、8类。原始重放最大分数差8.88e-16、B差4.44e-16、top1差0。匹配/重放6.80秒，训练步0。没有重选epoch、阈值或随机匹配种子。

| 数据 | seed | original−swapped净差 | ΔOA pp | ΔBA pp | ΔNLL | ΔBrier |
|---|---:|---:|---:|---:|---:|---:|
| sourceval |17|+28|+3.060|+2.767|−.01981|−.01017|
| sourceval |29|+24|+2.623|+2.720|−.01982|−.01009|
| sourceval |43|+24|+2.623|+2.620|−.02174|−.00923|
| test |17|+23|+.485|+.547|−.01162|−.00399|
| test |29|+23|+.485|+.522|−.01635|−.00698|
| test |43|+42|+.885|+.802|−.01141|−.00407|

test的B OA/BA分别41.3575/42.6882、41.9688/43.1215、41.2099/42.3125；original为42.6644/42.2600、43.6341/42.9272、42.3482/41.9368；swapped为42.1796/41.7129、43.1492/42.4053、41.4629/41.1346。original相对B净+62/+79/+54，swapped仍+39/+56/+12，两者相对B的BA均负。不能把交换前后差额当作可相加分离的“校准贡献百分比”。

sourceval original相对B净+22/+34/+16，swapped为−6/+10/−8。sourceval原来参与过头选点，正信号不等于独立确认；去重816对象后original−swapped净+25/+23/+20，仍为同数据描述。test逐折净差：seed17为+22/0/+1，seed29为+21/+4/−2，seed43为+21/+4/+17。fold1 seed17选中epoch0，故该折输出完全不依赖矩，不应把零作用写成信息不可用。

## 匹配覆盖与距离

同fold/seed/role、B预测类、validVV/VH，且不同MMSI；按源OOF train原始8维B各类尺度标准化距离做确定性最短合法边贪心，两两互换整块[pol,class,sigma/kappa]。标签只在冻结配对后用于报告。所有双向互换/一一使用/缺pol/未匹配原样检查通过。

| 数据/seed | 匹配对象/总数（比例） | 匹配MMSI | 距离p50 / p95 / max | 事后同类率 |
|---|---:|---:|---|---:|
| sourceval17 |890/915（97.27%）|616|1.266 / 3.007 / 6.472|41.12%|
| sourceval29 |894/915（97.70%）|618|1.268 / 3.228 / 8.234|38.70%|
| sourceval43 |890/915（97.27%）|617|1.331 / 3.044 / 5.739|39.55%|
| test17 |4724/4744（99.58%）|2967|.994 / 2.430 / 9.371|39.67%|
| test29 |4728/4744（99.66%）|2967|1.012 / 2.484 / 10.692|40.19%|
| test43 |4714/4744（99.37%）|2963|1.056 / 2.524 / 8.817|39.37%|

sourceval匹配890/894/890次折暴露，对应796/800/795唯一观测；每组均覆盖全部8类和32pass。test分别2362/2364/2357对，全部8类/85pass有匹配；同pass率5.25%/6.35%/5.73%，同港28.92%/30.20%/28.77%。高覆盖不等于近乎相同条件：距离尾部较大，多数donor异类、异港、异采集。整块矩的经验边际分布不变，不等于它与B/观测条件的联合分布不变。

未匹配sourceval 25/21/25条：单例条件组4/5/6，其余21/16/19因合法候选已被前序贪心使用；test20/16/30条：单例0/2/0，其余20/14/30为候选已用。没有“组内完全无不同MMSI候选”的非单例残余。未匹配保持原样。matched-only test ΔOA为+.487/+.486/+.891pp，ΔBA+.549/+.525/+.812pp，稀释很小；没有据距离事后过滤或重配。

## 配对区间与类别

pass/MMSI各2000次，source/test分开，条件于固定配对/预测，不重做配对或选点。test original−swapped：

| seed | OA pass CI / MMSI CI（pp） | BA pass CI / MMSI CI（pp） |
|---:|---|---|
|17|[−.171,1.181] / [−.270,1.221]|[−.102,1.321] / [−.115,1.246]|
|29|[−.327,1.273] / [−.398,1.278]|[−.561,1.511] / [−.417,1.403]|
|43|[.028,1.839] / [.042,1.763]|[−.029,1.677] / [−.060,1.715]|

test三seed NLL/Brier差的两种区间均为负；sourceval两种BA区间均正，OA seed43的pass区间跨0（[−.249,5.378]），MMSI不跨。sourceval pass bootstrap有87/2000次遗漏至少一类，只从剩余抽样取BA区间；其他全量区间对应缺类次数为0。所有细节及original−B/swapped−B区间见paired_intervals.csv，不挑块单位。donor可跨block，相互交换带来的跨块依赖及重新匹配不确定性未完整建模，区间为描述性条件区间，非因果显著性检验。

| 类别 | sourceval折暴露 | test对象 / MMSI | test original−swapped净差17/29/43 |
|---|---:|---:|---|
| bulk |193|1290 / 938|+7 / +7 / +13|
| fishing |60|145 / 130|+1 / 0 / +2|
| general |62|439 / 334|+6 / +6 / +3|
| product/chemical |339|1480 / 768|+3 / −6 / +13|
| container |86|519 / 325|+5 / +12 / +7|
| crude |79|405 / 220|0 / +7 / +3|
| tug |34|202 / 122|+2 / −2 / 0|
| offshore |62|264 / 136|−1 / −1 / +1|

class_intervals.csv保存96条逐类source/test×3seed×8类×2块区间（OA/BA在单类下均为召回，另含NLL/Brier）。test逐类召回区间均包含或触及0，不能包装为已定位稳定受益类别。类别点值也非只有原油轮：bulk/general/container三seed正；product/chemical和tug/offshore有负或零。source对应净差和全部匹配MMSI/类组成另见stratified_metrics.csv、composition.csv。港口亦混合，如Richards Bay +10/+6/+12，Fujairah −7/−5/−3，不支持处处有效。

## 四对原图已实际查看

固定seed17、test已匹配首ID。**rescued/harmed按original→swapped定义，非B→original。** 四query与四实际donor为8个不同对象，query全部来自Busan同一pass，因此不是8个独立场景；标题/CSV保留双方完整ID、真类、B/original/swapped预测、原尺寸/flags与分差。派生图采用matched_rescued.png、matched_harmed.png、matched_CC.png、matched_WW.png短名，原图未改。

- **借用后救回3504←7484**：两者均bulk，双路64窗，B均错为general。query为向右下的连续亮条、周围纹理较亮，Fujairah donor长条朝向不同、周围较暗。query本图矩仍错，借用后margin由−.19322变+.02082，确是借用救回；donor从+.23455降至+.02773仍对。并非每个对象都应以本图矩获益。
- **借用后误伤4708←returned_20260915_0001368**：query fishing，donor实际bulk，但B都预测fishing；双方VH无效，仅作原图展示、不进入矩。query VV为较长竖亮块，donor为紧凑小亮点且右下另有亮区。距离2.9868高于test17总体p95，不能称严格同条件。query margin+.19937→−.23963，转为general；donor一直错为fishing，margin进一步下降。不能由此把某亮区命名成误伤机制。
- **CC3494←32742**：同bulk、B均正确。query64窗近横向白亮条，donor VV72/VH80为近竖向且亮度分布不均的细长响应。query优势2.11142→1.72761，donor1.31807→1.47950，双方仍对；不同可见外观下矩变化不必造成top1改变。
- **WW3594←returned_local_knowledge_0012526**：query bulk，donor crude，B均预测crude；双方呈近竖向长亮区，周围纹理有差异。该例fold1/seed17是**epoch0零头**，双方交换前后分数完全不变，query一直错、donor一直对。没有信息作用，不能拿这例解释矩的好坏。

这些图只展示真实query/donor及模型分差，非κ像素贡献/因果遮挡。核对整组矩和固定head的作用，不从朝向、亮度、条纹推导物理机制。

## 中心化检查及研究边界

centering_check.csv记录各fold/seed的source/evaluation逐行类均值范围、所有合法边距离变化和donor变化。源均值范围最大.215068，评价集最大.103974；距离最大差.162248。sourceval改变donor行数按seed为11/32/23，test为228/275/349，共918条折暴露。确实影响配对，不能宣称等价。centered_correction为唯一授权修正，不按两版test孰优选规则。

已读主流程literature_comparison/CONDITIONAL_IMPORTANCE_NOTE_ZH.md：Strobl等2008正文2.4–2.5讨论条件重要性及树网格内置换，本实验并非其算法，也不是从真实条件分布采样，不能移用其统计保证；本次没有另行检索/自称精读2020论文。主流程另报score_strength某些普通弱化校准在test同时优于free OA/BA、source无同样双赢，这不否定本诊断的条件性效用，也意味着它不等于free是最佳部署模型。

结论限于当前固定模型和近似条件匹配。既不能把增量都说成纯校准，也不能把原始矩优势称作全独立于校准的因果证明。后续只执行已授权centered修正；本目录uncentered数据保留。无新训练、无Git、未改父模型。
