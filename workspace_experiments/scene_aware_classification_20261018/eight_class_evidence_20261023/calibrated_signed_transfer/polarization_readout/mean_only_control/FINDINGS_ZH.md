# Mean-only：同辅助 W 的 GAP 均值控制

## 结论

9 头完成。普通 GAP 均值在强 B 上有明确 OA/NLL/Brier 收益；局部分布 local 仍保留三个 seed 更高的 BA，但 OA 均低于 mean-only。本轮不能称 local 无效，也不能称 mean 全面优；更不能继续声称局部分布全面优于同 reader 的均值信息。

mean-only 相对 B 测试净正确数 +77/+50/+84，OA +1.6231/+1.0540/+1.7707pp，BA −.1987/−.3184/+.4034pp。local 相对 mean 的 BA +.8253/+1.5500/+.1425pp，但 OA −1.0750/−.1897/−1.5388pp。mean 的化学品船收益较大，local 更保留原油和海工补给类的表现；这个取舍不能只靠总 OA 或 BA 命名为好坏。

源验证与测试均有两者各自独对和独自救回 B 错的对象，为下一轮提供合理但未验证的互补路径。未构造测试 oracle、未训练融合、未搜索力度或新增臂；既有 local/extra_score/B 完全保留。

## 方法、数据与检查

使用原 signed_features.npz 的 mean_pol[15,15251,2,8]，以同辅助 W、原 native160/footprint/C4 每极化均值复制为两槽，输入与 local 相同 11 维 descriptor、209 参数类共享头：

```
m_pc = C4avg(footprint_weighted_mean_p(W_c,pol F_pol,p))
slots[...,0] = slots[...,1] = m
x_c = [center(B)_c,
       m_VV,c, center(m_VV)_c, m_VV,c, center(m_VV)_c,
       m_VH,c, center(m_VH)_c, m_VH,c, center(m_VH)_c,
       validVV, validVH]
z_c = (x_c - mu_source) / sd_source
r_c = Linear16to1(GELU(Linear11to16(z_c)))
score_c = B_c + r_c - mean_class(r)
```

两槽不是 sigma/kappa，不称物理 moments；该头不乘 κ、不 tanh。两重复槽保持名义架构和初始化，但有效自由度不完全等价。均值分支还是含 B 的监督残差读出，不是“直接用 auxA 替代 oldAD”。

- 只新增 3 fold×seed17/29/43，共 9 个头、1800 步。AdamW .003/.01、jointCE=.5普通+.5八类均衡、每5步 sourceJ 选择含0/精确平局最早，全部沿用。9 个选点落盘后才评价测试标签，0 个选择0步。
- train 4392/4419/4418；val 310/286/319，共915折暴露、816唯一对象；test1561/1632/1551，共4744唯一对象、2973MMSI、18港、85pass。每 seed 独立统计，不能把种子重复当独立数据。
- mean_pol 维度正确、有限；全 cohort 有效 VV15251、VH13783，invalid 原值严格0。源内 scaler 只本折 train，按原规则转换 invalid0 而不另作后置抹零。
- 15 reader 的 means.sum(pol)+intercept 重放原辅助 A/OOF，最大误差 1.6951e−7，top1变化0。cache/cohort/fold instance_id 对齐；每 inner fit/held 属本折 train，query MMSI 不相交，held 恰一次覆盖 train，outer fit 精确等于 train，测试行无 fit 暴露。见 data_checks.csv/data_summary.json。
- 冻结 B=T(oldAD) 相对 strong_calibration_increment 已存逐分数最大差 1.7764e−15，top1变化0。按原 local 构造函数和 seed 重建初始 state、逐张量 exact 相等，本轮初始 state 存入每个 checkpoint；不是拿已训练 local 参数当初始化。零末层严格回 B，保存选点重放通过。
- 源训练用 T(oldAD_OOF)+mean 的独立合法 inner held 读出；T 已在这批源标签上训练，明确不是二级或全栈 OOF。标准化没有 val/test 进入，T 与新增头均使用源验证选点，源验证不是独立确认。旧 AD 与辅助 W 的 inner 包不同不自动构成泄漏，各自 query 身份排除即可；不声称整链严格嵌套。

## 测试与源验证结果

以下 mean-only 相对冻结 B；OA/BA 为%，差为pp。pooled.csv 同时保存 B 的全部指标。

| 集合 | seed | 净数 | OA | BA | ΔOA | ΔBA | NLL | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| test |17|+77|49.9789|41.3464|+1.6231|−.1987|1.404706|.656068|
| test |29|+50|50.1897|41.7782|+1.0540|−.3184|1.392372|.648732|
| test |43|+84|49.3887|41.0511|+1.7707|+.4034|1.436829|.663485|
| sourceval |17|+27|57.8142|48.1713|+2.9508|+1.3411|1.322338|.606733|
| sourceval |29|+11|58.3607|48.2136|+1.2022|+.3501|1.315054|.599337|
| sourceval |43|+18|55.9563|46.8380|+1.9672|+.8263|1.337031|.610817|

全部对照只读既有预测，没有重训。下表均为 mean-only 减参照。

| 集合/参照 | seed | ΔOA | ΔBA | ΔNLL | ΔBrier |
|---|---:|---:|---:|---:|---:|
| test/local |17|+1.0750|−.8253|−.071677|−.027537|
| test/local |29|+.1897|−1.5500|−.021198|−.009889|
| test/local |43|+1.5388|−.1425|−.043182|−.016858|
| test/extra_score |17|+1.3912|+.0858|−.053654|−.021312|
| test/extra_score |29|+1.2226|+.9157|−.054113|−.020725|
| test/extra_score |43|+1.1594|+.2120|−.066749|−.017371|
| sourceval/local |17|+2.2951|+.7031|−.077928|−.035446|
| sourceval/local |29|+.9836|−1.2081|−.027436|−.011856|
| sourceval/local |43|+1.0929|−.2583|−.050128|−.022342|

源 val 的 mean OA/NLL/Brier 路径已正，不是由测试结果选出来。相对 local 的 BA 在源端也是取舍而非一致优势；完整 sourceval 对 extra_score 见 paired_metrics.csv。T 非二级 OOF、验证选择复用和类别组成均是适用边界，现有结果不能把任何一项单独定为差异的根因。

## 配对置信区间

固定预测的 pass/MMSI 聚类2000次 bootstrap，seed20261023；不重拟合/重选点，不跨 seed 合并，不包含整条训练流程不确定性，未作多重校正。下列 pass95%CI；MMSI 同存 paired_intervals.csv。

| 比较(mean减参照) | seed | ΔOA [CI] | ΔBA [CI] |
|---|---:|---|---|
| B |17|1.623 [.933,2.361]|−.199 [−.892,.499]|
| B |29|1.054 [.244,1.875]|−.318 [−1.059,.510]|
| B |43|1.771 [.917,2.596]|.403 [−.464,1.344]|
| local |17|1.075 [.351,1.770]|−.825 [−1.575,−.099]|
| local |29|.190 [−.813,1.184]|−1.550 [−2.561,−.507]|
| local |43|1.539 [.777,2.267]|−.143 [−.889,.626]|
| extra_score |17|1.391 [.639,2.135]|.086 [−.657,.848]|
| extra_score |29|1.223 [.476,1.967]|.916 [.213,1.610]|
| extra_score |43|1.159 [.238,2.062]|.212 [−.811,1.305]|

mean 相对 B 的 OA 三个区间均正，BA 均跨零。local 相对 mean 的 BA 前两 seed pass区间正、第三跨零；seed17 在 MMSI 口径下也跨零。不能把相关的三 seed 或两种聚类口径当独立重复证据。

## 类别、港口与有效极化

测试相对 B 的净正确数，均为seed17/29/43。local列只读原结果。

| 类别 | n | mean-only | local |
|---|---:|---|---|
| bulk |1290|+32/+43/+20|+24/+40/+12|
| fishing |145|−2/−2/0|0/0/0|
| general |439|+9/+7/+10|+3/+8/+5|
| chemical |1480|+64/+28/+61|−10/−29/−19|
| container |519|−5/−6/−5|−10/−8/−9|
| crude |405|+2/−3/+21|+17/+24/+22|
| tug |202|+1/+3/−2|+1/+2/−1|
| offshore |264|−24/−20/−21|+1/+4/+1|

mean 的 OA 增益主要来自 bulk/chemical/general，offshore 有持续明显代价（召回−9.09/−7.58/−7.95pp）。local 对 offshore 和 crude 的保留解释其 BA 相对优势的重要部分，不是“BA小降所以均值无效”。源 val mean 的 chemical净+20/+6/+12、offshore−2/−3/−3；local对应 chemical−2/−12/−2、offshore−1/+1/+1，已可见类似类别取舍。源 mean crude +6/−3/+4 与 local +5/+7/+4 则不是每seed同样差异。

mean 测试港口净数：Port Klang +23/+34/+25、Santos +14/+11/+7、Singapore +22/+10/+13；Jebel Ali +5/−11/+14、Los Angeles −1/−6/+6 不稳定。完整18港、fold、pass/product见 port_metrics.csv、fold_metrics.csv、stratified_metrics.csv、acquisition_metrics.csv；不能把港口相关性称物理因果。

有效VH4436例的 mean净+76/+49/+80，缺VH308例+1/+1/+4；sourceval对应833次+26/+12/+18、82次+1/−1/0。完整 OA/BA/NLL/Brier 在 vh_metrics.csv，不把观察到的有效组收益解释成 VH 的因果贡献。

## 互补计数：只描述，不构造 oracle

local独对指 local对且mean错；mean独对反之。救回仅在B错中计，损失仅在B对中计。下表 L/M 为 local/mean；sourceval仍为915折暴露，不把816唯一对象伪称915独立对象，按首fold去重的敏感性表另存 overlap_validation_unique.csv。

| 集合 |seed|L独对|M独对|共同救回|仅L救回|仅M救回|共同损失|仅L损失|仅M损失|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sourceval |17|19|40|11|6|35|6|5|13|
| sourceval |29|24|33|15|13|17|10|16|11|
| sourceval |43|20|30|13|6|22|3|8|14|
| test |17|123|174|37|31|157|25|17|92|
| test |29|141|150|63|77|93|42|57|64|
| test |43|120|193|48|34|153|31|40|86|

逐类独对（seed17/29/43）；全部逐类救回/损失交叠、B对错分母已保存在 overlap_classes.csv，共48行，未跨seed当独立样本汇总。

| 类别 | source L独对 | source M独对 | test L独对 | test M独对 |
|---|---|---|---|---|
| bulk |7/4/5|6/8/6|30/36/33|38/39/41|
| fishing |0/0/0|0/0/0|2/3/1|0/1/1|
| general |1/0/1|1/2/3|4/10/8|10/9/13|
| chemical |4/3/6|26/21/20|21/15/27|95/72/107|
| container |2/3/4|2/2/1|11/15/9|16/17/13|
| crude |2/10/0|3/0/0|24/33/18|9/6/17|
| tug |1/0/0|1/0/0|4/2/2|4/3/1|
| offshore |2/4/4|1/0/0|27/27/22|2/3/0|

可检验路径不是由测试 oracle 得出：源端 chemical 已有 mean 独对26/21/20与local独对4/3/6；offshore有local独对2/4/4与mean1/0/0，crude主要是seed29的local10 vs mean0。既有 local 的独对并非都来自独有救回，例如测试offshore的local独对27/27/22中，仅local救回1/7/3、而仅mean误伤26/20/19。应同时考虑保护已有正确对象与新增救回，不将两者混成“发现新证据”。这些数只能说明两种读出的错误不完全重叠，不能证明存在无需标签即可辨认的路由条件、互补信息独立或联合训练一定涨点；本轮不实施联合模型。

## 四张原始 VV/VH 实际回看

固定seed17，每结局首instance_id，4图均已view_image。短名 panels/mean_rescued.png、mean_harmed.png、mean_CC.png、mean_WW.png；完整ID/原路径/valid/窗口/最终竞争类在review_cases.csv，实际两槽mean在review_inputs.csv。未根据图像改模型或标签。

| 结局/ID | 可见事实 | 实际读出与边界 |
|---|---|---|
| 救回 pair_0005030，general/Busan | VV64中心斜向亮体并有细直亮线，VH64的框内亮团更分离；各自160pad的有效覆盖.160 | bulk→general，真类对最高竞争类margin −.15439→+.34805。相同对象既有local也救回（+.06857）；普通均值已足以修此例，不能把该案例作为局部分布独有证据。 |
| 误伤 pair_0007310，bulk/Fujairah | VV/VH72均为斜向连续强亮体，周边纹理存在，两路有效覆盖.203 | bulk→crude，margin +.02655→−.09430。bulk残差约+.89但crude约+1.01，竞争差异造成误伤；真类得分上升不等于预测改善，外观不能决定物理原因。 |
| CC pair_0003494，bulk/Busan | VV/VH64横向亮体；VV周边灰度纹理更强，VH亮体有局部不均 | bulk保持正确，margin2.35965→2.83324；local也CC，不能将大分差增加当新增正确数。 |
| WW pair_0003504，bulk/Busan | VV/VH64斜向长亮体，VV周围纹理更明显；不据外观判断标签或物理成因 | general仍胜，margin−.64017→−.76427；mean使分差更差，local原为−.55098但仍WW。展示分数方向差异，不证明某一物理机制。 |

图题为 GAP means重复槽的直接残差，不误称κ贡献；原分辨率与160crop/pad分开显示，绿色框仅记录OBB，footprint参与池化而非遮原图。卷积感受野仍可混合padding/框外内容，不把均值或局部分布叫纯目标/纯海面证据。

## 成本与收口

9头1800步，训练/源选择/保存/重放累计10.45秒；缓存加载、针对性检查与评价合计15.19秒。无reader拟合、T拟合或图像提取；此计时不是新增部署推理成本，未测完整端到端或本轮单样本推理时间。相同209参数并不使mean/local有效自由度和上游计算成本完全相同。

交付包括9模型、369曲线点、全部选择/check/数据重放记录、逐样本/pooled/类港VH采集/配对CI、交叠逐类表和4实际查看面板。原有效local与旧T不动，未Git。保留两种读出各自收益与边界，到此结束，不追加训练/融合/搜索。
