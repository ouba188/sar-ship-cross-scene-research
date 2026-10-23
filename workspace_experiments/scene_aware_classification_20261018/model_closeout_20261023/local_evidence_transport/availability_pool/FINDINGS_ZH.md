# 固定参考可用性：有界源端诊断

## 结论与建议

结构缺失与真实零/低匹配确实被当前`mean(2)`混在一起，但**不应据此转向大量新头训练或把它作为核心创新**。utility_train中真类仅stem两条、layer1一条发生“有效query配一条有效参考、一条结构无效参考”；主竞争类描述margin实际改变的query分别仅12/644、7/644。另一方面，全类别候选可用性不一致并不罕见：stem405/644、layer1 147/644条，主要涉及非真类fishing候选。

保留valid-reference mean为明确的预处理候选及可重放缓存即可。本轮不训练、不做旧头输入替换精度宣称，不据外部189调规则，不扩大为新的模块研究。若主流程日后需要验证，保持相同refs与语义拒配，做匹配的训练/源选点；当前没有收益证据要求立即启动。

## 范围与算术

读取当前`match_readout.py`、`train_readout.py`，只重算stem/layer1的pooled、independent、anchored、body_match。PCA、两条固定refs和运输参数全部冻结。处理3387条源观测的无标签结构信息，**标签诊断仅utility_train644条**，竞争类固定为seed17 full的最高非真类；不把三个seed视为独立样本。未读取外部缓存/标签/成绩，未训练、下载、提取骨干或改父代码/Git。

对每参考原输出`v_k=(sum T*sim, sum T)`，结构标记`a_k`仅由有效极化及正区域像元质量决定。anchored side/end另需query和reference各自body存在；body_match的side/end为设计屏蔽，不计为缺失。

- 旧值：`(v_1+v_2)/2`。
- 可用参考均值：`sum(a_k*v_k)/sum(a_k)`；分母0则输出0，另保存计数。
- **不除以语义匹配质量。** 一个真实可用但被null拒配的参考仍进入分母。没有补第三参考、改变参考排序、增加neff门槛或重做运输。

两层四臂的旧均值对父缓存最大误差均0；结构无效参考原输出最大绝对值0。两参考可用时新旧完全相同，一参考可用时新值严格为旧值两倍，无参考时为0。语义拒配仍存在，例如所有源候选中anchored有效参考的最小真实匹配质量stem约1.93e-5、layer1约.003，并未归一为1。

## 每query跨候选类可用性

utility_train有644个VV可用query、561个VH可用query，另83个无有效VH。对这644个query，三个区域的以下计数相同；**不能将body/side/end重复当作三倍独立证据**。本数据中pooled/independent/anchored结构计数一致，代码仍明确保留anchored对body的额外要求。

|层/极化|至少两个候选类的可用参考数不同的query|有某候选类仅1条参考的query|仅1条参考的query×候选类对，每区域|有效query却某候选类0参考的类对|
|---|---:|---:|---:|---:|
|stem VV|0/644|0|0|0|
|stem VH|405/644|405|446|35|
|layer1 VV|0/644|0|0|0|
|layer1 VH|147/644|147|179|0|

每个query的8个候选类计数均在`utility_query_availability.csv`；83个VH缺失query对所有类都是0，不计为跨类差异。上表的35个零参考类对属于stem：fishing1、chemical34；没有因为分母修正而补充它们的证据。

|候选类|stem图库有效VH/8|layer1图库有效VH/8|stem有效query选到1参考的次数/561|layer1对应次数/561|
|---|---:|---:|---:|---:|
|bulk|7|8|1|0|
|fishing|4|5|364|117|
|general|6|7|1|6|
|chemical|6|7|7|28|
|container|8|8|0|0|
|crude|7|8|36|0|
|tug|6|7|0|0|
|offshore|5|6|37|28|

真类一参考query仅stem的obs11591(fishing)、obs31024(chemical)，layer1仅obs31024。因此不能把大量非真类候选的缺失频率写成大量真类证据被削弱。

## 真类减固定full竞争类的描述变化

以下仅为VH相似度质量的**输入描述margin**，平均分母均为644个utility_train query；不是新模型logit、OA/BA或因果效应。VV变化全0。body_match只保留body。

|层/臂|body平均变化|side平均变化|end平均变化|body增加/减少/不变query|
|---|---:|---:|---:|---|
|stem pooled|-.002556|-.006883|-.007158|3 / 9 / 632|
|stem independent|-.004014|-.006832|-.007065|1 / 11 / 632|
|stem anchored|-.004014|-.005230|-.001905|1 / 11 / 632|
|stem body_match|-.004014|不适用|不适用|1 / 11 / 632|
|layer1 pooled|-.000371|-.000987|-.001458|3 / 4 / 637|
|layer1 independent|-.000747|-.001367|-.001433|2 / 5 / 637|
|layer1 anchored|-.000747|-.001077|-.000411|2 / 5 / 637|
|layer1 body_match|-.000747|不适用|不适用|2 / 5 / 637|

同一query的真类及竞争类都可能改变，不可把计数相加。stem最高非真类一参考11条，与真类2条有重合，共12条margin变化；layer1竞争类6条，真类1条，共7条。均值并非普遍改善：对于当前full已正确的347条，anchored body在stem为0增/10减、layer1为1增/5减；对full错误的297条，分别1增/1减与1增/0减。变化方向也不是简单“去除缺失项总会增加真类优势”。

逐真类、body/side/end、相似度质量/匹配质量及full对错分层保存在`utility_descriptor_margins.csv`；只作源端描述，不据此选新的模块或力度。

## 一例源原图与实际参考：已view_image

已实际查看`source_31024.png`的完整2×5面板：query31024、真类参考24840/26391、固定full竞争类参考13344/30434，VV/VH分别展示原生160裁补，没有改变图像尺度、换供体或挑最大的正变化。它是layer1唯一真类一参考的utility query，故不存在再按漂亮程度选择。

- query31024是真chemical，full预测offshore，固定分差为-.640698。可见query VH目标较暗，真类参考24840和两个竞争参考的VH亮体更清楚；这些外观不足以给出物理类别原因。
- 真类参考26391虽有原PNG，但已保存的局部VH输入标记无效，图中明确标注；**文件存在不等于可用局部证据，也不能把无效输入当成观测到的低匹配**。未改其有效标记或追溯扩展提取规则。
- 真类有效参考24840的VH body原相似度质量为.123917、真实匹配量.499013；另一参考两者均为0且结构无效。旧均值为.061959/.249506，新均值为.123917/.499013，语义拒配后约一半质量仍保留，并未变成全匹配。
- 两竞争参考都有效，其body相似度质量均值.118007不变。因此真类减竞争类的VH body描述margin从-.056048到+.005911；side从+.089978到+.163787，end从+.009201到+.017446。

这说明算术上确有可解释变化，**不是分类救回**。本轮没有重训或用旧head替换输入，full原分差不会自动随这些描述变化。不能由这一例声称主模块损失都来自缺失，也不应把它作为新核心贡献。

## 文件与停止点

- `diagnose_availability.py`：ASCII代码，固定PCA/refs、逐参考重算与源统计；约14.36秒。
- `per_reference_stem/layer1.npz`：四臂原始分数`[N,C,K,P,R,2]`、每臂结构availability、refs、全局indices/instance_ids及区域存在标记，支持以后公平匹配训练。
- `scores_current_stem/layer1.npz`、`scores_valid_stem/layer1.npz`：旧/新均值`[N,C,P,R,2]`及每臂可用参考数；原父缓存未覆盖。
- `structural_coverage.csv`、`bank_coverage.csv`：各源角色无标签覆盖，及utility真类覆盖；非utility角色未做标签效果分析。
- `utility_query_availability.csv`、`utility_query_summary.csv`、`utility_descriptor_margins.csv`、`utility_one_reference_cases.csv`：主诊断与逐query追溯。
- `complete_source_case.py`：ASCII代码，沿用现有native/区域绘图接口，正确标成SOURCE和固定B竞争类；`source_31024_evidence.csv`及`source_31024_images.json`记录实际参考、路径、有效性与分数。
- `PROTOCOL_ZH.md`、`diagnostic_summary.json`、`source_index.csv`、`run.log`：协议和回执。

**本支线收口，不启动训练，不继续扩大诊断。** valid-reference mean保留为修正预处理候选，主模块发展不应围绕这一个真类源案例转向。所有进程已结束。
