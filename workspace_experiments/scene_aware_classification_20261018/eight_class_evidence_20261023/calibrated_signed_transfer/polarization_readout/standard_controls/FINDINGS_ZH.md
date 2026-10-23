# 标准极化读出控制：18轨迹完成

## 结论

当前证据不支持条件κ载体机制是必要的。209参数free_shared不乘κ、不tanh，三seed相对B净正确数为**+25/+1/+22**，相对原real为**+13/+3/+11**。但free并非OA/BA全面占优，seed17的BA比B、real和capped均低。不能把点估计比较写成统计上确立的全面替代。

16参数static_pol相对B净**-7/+9/-3**，BA三seed均有小正；说明静态类×极化校准也能形成部分收益，但不足以稳定替代更丰富的读出。static不使用B/σ descriptor且容量小，不能将其与real/free的差额全部归于逐图适应性。

原27头及结果完全保留。原real/capped与同容量aggregate的比较，仍是路别信息的原有证据；简单读出表现相近或更好不会抹去它。另一方面，本轮没有free自身的aggregate匹配臂，free输入又包含B、σ、κ，故其对B的收益不能单独证明路别κ的因果贡献。不追加臂或调参。

## 固定执行

先写PROTOCOL_ZH.md，再训练3fold×3seed17/29/43×2arm，共18头/3600优化步。直接复用父run_polarization的inputs/load_pol及基线、jointCE、exact_j函数，source train标准化、inner-held矩/A_OOF、冻结C得到B均保持一致。每5步含0按source J选最早点；全部18选点保存后才统一test。

free_shared为11→16 GELU→1，共209参数，直接去除类公共输出；static_pol为8×2系数tanh后乘真实κ，共16参数，无逐图gate。两者均零初始化回B。原候选226参数，不能称严格等容量；static带固定类索引，free在类间共享映射，对称性也不同。

三个0步选点：static fold0 seed17、static fold2 seed43、free fold1 seed43；仍各训练200步。没有提取新特征、重拟合Reader或重训原候选。static初始系数都为0，本身无随机初始化，但各seed的冻结C/B不同，不将它们当独立数据重复。

## 源验证与测试

validation为915次折次暴露/816唯一观测；test为4744互斥观测。下列源验证是各折选点合并后的描述，不是独立泛化估计。另保存最小fold去重的validation_unique_metrics.csv。Δ均对相应seed冻结B。

|臂|seed|val ΔOA pp|val ΔBA pp|val净正确|test OA|test BA|test净正确|test ΔOA pp|test ΔBA pp|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|free|17|+1.420765|+0.314164|+13|45.088533|38.341979|+25|+0.526981|-0.367509|
|free|29|+0.437158|+0.539450|+4|47.344013|38.787783|+1|+0.021079|+0.281912|
|free|43|+0.218579|+0.002468|+2|47.407251|39.075708|+22|+0.463744|+0.430150|
|static|17|+0.437158|+0.828157|+4|44.413997|38.833509|-7|-0.147555|+0.124021|
|static|29|+0.655738|+0.317799|+6|47.512648|38.880659|+9|+0.189713|+0.374788|
|static|43|+0.218579|+0.207980|+2|46.880270|38.787589|-3|-0.063238|+0.142031|

test救回/误伤：free为123/98、43/42、56/34；static为14/21、33/24、9/12。free的test NLL/Brier分别1.481222/0.692739、1.434987/0.676868、1.441190/0.677272；static分别1.517884/0.703993、1.435708/0.677782、1.444964/0.679622。完整balanced NLL/Brier保留在pooled.csv，不仅报告top1。

## 匹配原候选

每行均为同fold、seed、instance的配对差，非另换基线。source validation与test均有matched_metrics.csv及逐类matched_classes.csv，以下仅test：

|控制臂|seed|对real净差|对real ΔBA pp|对capped净差|对capped ΔBA pp|
|---|---:|---:|---:|---:|---:|
|free|17|+13|-0.452508|+1|-0.490681|
|free|29|+3|+0.130908|-3|+0.260790|
|free|43|+11|+0.175171|+17|+0.152653|
|static|17|-19|+0.039023|-31|+0.000850|
|static|29|+11|+0.223784|+5|+0.353666|
|static|43|-14|-0.112948|-8|-0.135466|

free相对real的NLL/Brier三seed均降低，但多数配对区间跨0。85个pass、2000次bootstrap下，free-real的ΔOA区间分别[-0.403,0.896]、[-0.262,0.361]、[-0.219,0.624] pp；ΔBA分别[-1.278,0.156]、[-0.132,0.402]、[-0.136,0.465] pp。不能据OA点估计三次为正宣称已证明机制优越。capped及B的配对区间、MMSI块区间全部保留，不做多比较选胜者。

free相对父aggregate的净差为+26/+5/+22，但这里参数/函数约束不匹配，只作描述。若要隔离free中的极化输入价值，需要另行授权的匹配研究；本轮未做、也不据此追加。

## 类别与覆盖

以下为相对B的test逐类净正确数，原类顺序不变：

|class|free17|free29|free43|static17|static29|static43|
|---|---:|---:|---:|---:|---:|---:|
|0 bulk|+27|-11|-12|-7|+4|-7|
|1 fishing|-2|+1|0|+1|+1|0|
|2 general cargo|+12|0|-2|+2|+3|0|
|3 chemical|+19|+3|+22|-6|-6|-1|
|4 container|-17|-2|0|-1|-2|0|
|5 crude|-6|+9|+14|+4|+9|+2|
|6 tug|+1|0|-1|0|-1|+1|
|7 offshore|-9|+1|+1|0|+1|+2|

free seed17的OA收益与bulk/chemical正确数增加并存，但container/offshore/crude损失使BA下降；不是所有类别均获益。seed29与43的类别转移又不同，不能从单seed挑一个统一图像原因。

有效VH4436、缺VH308保持不变。free在有效VH子集净+26/+2/+25，在缺VH子集-1/-1/-3；static分别-7/+12/-4和0/-3/+1。此为观察分层，类/港口组成不同，不能单凭这些数证明双极化信息的独立因果价值。逐fold、类、VH、pass、港口保存在stratified_metrics.csv。

## 八张原图实际回看

固定seed17、每臂每种结果首ID，8张均用view_image查看，未改变采样来迎合机制。标题正确区分free直接残差与static固定系数κ读出。原VV/VH、实际160 crop/pad、有效性、旧/新最高竞争类均可追溯。

1. free救回pair_0003594，bulk/Busan，两pol88窗。两路纵向亮目标及周边纹理可见；预测5→0，分差-0.69374→+0.04616，直接残差改变+0.73989。可见目标不等于已识别物理类别机制，这里不是κ门控贡献图。
2. free误伤pair_0003555，bulk/Busan，两pol64，斜向亮目标；0→3。旧最近竞争类2变成3。对旧类2的分差反而提高+0.26269，但对实际最强竞争类的真类margin从+0.04473到-0.01402，说明不能用固定旧r的改善掩盖新赢家误伤。
3. free持续对pair_0003494，横向亮目标，0仍正确，margin +0.38865→+0.88162；持续错pair_0003504，斜向亮目标，仍预测3，margin -0.96513→-1.83519。无κ载体上界的自由残差可同时强化已有正确证据和错误证据，不能只展示救回。
4. static救回pair_0043352，fishing/Ningbo-Zhoushan，两pol64，框内响应较暗、VH有较显著小亮斑；0→1，margin -0.10895→+0.02735。固定系数仍能产生逐图不同残差，因为κ随图变化；“无逐图gate”不等于输出常量。
5. static误伤pair_0032880，bulk/Ningbo-Zhoushan，两pol64，细长分段亮目标；0→2，margin +0.10653→-0.00838。与救回相同fold1/seed17系数，差别来自输入κ及B，不能从外观强称极化噪声被正确/错误抑制。
6. static的CC/WW首例同为pair_0003494/0003504，所属fold0/seed17选0步，分差分别维持+0.38865和-0.96513。原图确已查看；这是fallback行为，不是“图中无可利用信息”的证据。

原始像素只用于回看，不参与test选点。8例不代表整体比例；没有逐像素配准乘积或物理散射解释。

## 验证、成本与文件

18个零头严格回B；保存后source验证J重放一致，float64最终推理top1对应同一J。与父real的B最大差1.78e-15、top1一致。source标准化和OOF读取直接复用父函数，无新坐标拟合。C非全栈OOF、阶段性复用验证、队列已开发暴露等限制继续成立。

18轨迹共3600步，CPU训练/保存/验证重放计15.47秒；run_controls总18.29秒，不含分析和人工图像查看。原27头未动，无新GPU提取、无Reader拟合、无额外训练。全部执行会话已退出。

PROTOCOL_ZH.md为训练前协议；run_controls.py/analyze_controls.py为独立脚本。models/含18头；selections.csv、selection_curves.csv、costs.csv、checks.csv、baseline_replays.csv记录选择与重放。predictions.csv/pooled.csv保留两角色全seed预测与概率质量；matched_metrics.csv、matched_classes.csv、paired_intervals.csv保留原real/capped/aggregate配对；static_coefficients.csv保存每类每pol的原参数及tanh系数。review_cases.csv和panels/对应8个已查看案例。

本轮到此结束，不因free表现较好删掉条件读出原结果，不将“载体机制必要性未成立”偷换成“极化信息无用”，也不因static较小而声称其与226参数候选容量等价。仅本目录写入，无Git。
