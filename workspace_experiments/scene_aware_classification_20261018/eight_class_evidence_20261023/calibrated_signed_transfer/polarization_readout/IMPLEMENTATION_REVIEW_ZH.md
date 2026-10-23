# 极化读出限定实现审阅

2026-10-23。结论：**当前版本未发现需重训或阻止结果分析的实际实现错误。** 审阅期间主流程补入final-validation J断言，并修正aggregate面板矩值来源；以下以这两个修改后的版本为准。本审阅未启动训练、未改主脚本、未选择新模型、未Git。

## 实际核对

|范围|检查与结论|
|---|---|
|OOF及原读出对应|`parent.load_fold`校验ID和原inner-held索引，`load_pol`复用同名reader与held，源train每行恰好覆盖一次。三折实际缓存的outer/OOF逐pol求和与原sum缓存逐值一致。没有用outer矩替换源训练OOF矩。|
|缺失极化|实际全缓存invalid槽矩为0；每样本至少一路有效。源train三折n=4392/4419/4418，缺VH=487/493/505。标准化后缺失槽descriptor可非零，这是固定缺失参照而非偷填真实矩；乘法carrier仍严格0。三臂共享valid flags。|
|aggregate总矩|`organized`以有效路数平均原sum矩，再仅放回有效槽。源train实际缓存逐值保持sigma/kappa总和，缺VH槽仍0；只有VV时不错误除以2。descriptor和乘法carrier同时替换，未保留隐藏的真实每pol载体。|
|门控与初始化|11->16->2/GELU，226参数；`tanh`逐路乘kappa，随后求和及类中心化。三臂同seed权重逐值一致，实际源输入下零头严格0。类别共享结构及来源标准化与协议相符。|
|冻结B|使用父同fold/seed的C及base/OOF A，三臂共用B。脚本对父increment缓存作B重放和top1检查。C非全栈OOF的继承限制未消失，但不是此次引入的错配。|
|选点隔离|jointCE、200步、AdamW .003/.01、每5步源val J、0步及严格大于更新均符合协议。27选点保存后才统一评价；没有test用于选点的路径。|
|输出重放|先前只重放float32 forward，现第201-204行已核对最终float64 parts路径的源val J。另以落盘CSV独立核对27头，最大J差5.55e-17；50931行VV/VH parts重构分数最大误差1.78e-15，存储分数argmax与prediction全部一致，缺VH parts最大绝对值0。无需放宽误差。|

以上运行仅为已有缓存的结构前向及输出核对，没有优化步骤。原`structural_checks`的缺VH/反号carrier测试也通过；没有扩展到无关防御审计。

## Capped 的准确含义

令每类两路kappa为k_p，S=sum(k_p)，L=sum(abs(k_p))，原始未类中心化增量u=sum(tanh(g_p)*k_p)。真实pol有 `abs(u)<=L`。capped乘 `abs(S)/max(L,1e-12)`，因此 `abs(u_capped)<=abs(S)`；分母低于阈值时只会进一步缩小，不会放大。

aggregate有m个有效槽，每槽carrier=S/m，因此 `abs(u_aggregate)<=abs(S)`。**capped与aggregate具有相同的每类未中心化幅度上界，而不是扩大幅度对照。** 类中心化后某个单类delta可能超出这个原始界，但两臂都接受相同类中心化；类间margin公共项抵消，不能把单类中心化值误当cap失效。

若S=0，capped该类未中心化贡献为0；其他类仍可变化，所以不等于该样本完全无法翻转。缺VH时L=abs(S)，正常非极小数值下cap因子为1。已保存的VV/VH parts是实际乘cap后的贡献，分析闭合应使用这些parts，不是原kappa直接相加。

若capped相对aggregate显示可信收益，可排除“必须扩大原始幅度上界才能获益”这一必要解释；不能据此证明反号抵消就是误分类根因。它仍保留逐路sigma/kappa、不同标准化和门控参数化，cap也改变梯度/优化与正则化。当前结果不应预写成“解除可达上界带来收益”，更不能说capped处处优于aggregate。已记录主流程更正：seed29 capped对aggregate的BA低约0.0528pp；不以更正前概括作结论。

## 对照能回答什么

real与aggregate有同226参数、相同B/flags、相同seed初始化/预算和数据对应，足以比较“保留逐极化矩的读出”相对“只保留总矩的同形状读出”是否带来净价值；缺VH标记和参数数量不是两者独有优势。

它不是只改变一项信息的纯因果等容量证明：real同时改变descriptor、carrier及未cap时可达幅度，且各臂按自身source输入拟合scaler。capped提供有针对性的幅度限制比较，但不是将real的训练轨迹原封不动搬到另一个上界。原signed113/std81/score280只能作为已冻结参照，不是226参数匹配控制。本试验不修复校准BA下降；A/B/旧A+D继续各用原协议。

## 分析脚本与已关闭问题

- `analyze_polarization.py`直接调用`parent.main('polarization_readout', ...)`可行，不受父CLI choices限制；父stage路径及非calibration时的B标签正确。
- 主比较按instance_id对齐real与aggregate/capped，三臂同B使net之差等于正确性之差；pass与MMSI分别计算固定模型配对区间。当前匹配CI包括real-aggregate和real-capped，各臂-B由父分析生成；不含capped-aggregate的直接CI，故后者不能冒用前两种CI或据点估计称显著。
- old/new最强错误竞争类分别由B与最终score计算，VV/VH各自的真类减竞争类parts相加后等于实际margin变化；类中心化公共量正确消去。竞争类切换有记录，不再只看旧竞争类。
- 先前复用renderer给aggregate显示原始真实每pol矩，存在解释输入的歧义。主流程已增加`show_used_polarization_moments=True`，aggregate显示sum/有效路数后的实际输入槽，并注明不是原始VV/VH矩；该问题关闭，不影响已训模型和真实parts闭合。
- 本轮未替主流程重新挑案例或审阅新结果图；原图实际查看、配对CI解释和最终效果报告由主流程继续。源码检查与实际输出重放已完成，到此收束。
