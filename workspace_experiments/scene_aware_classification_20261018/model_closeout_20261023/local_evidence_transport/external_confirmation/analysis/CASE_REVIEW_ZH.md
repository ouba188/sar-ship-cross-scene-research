# 外部3075观测：实际query与源参考图审

日期：2026-10-23。已运行 `external_cases.py`，逐张查看6张面板的VV/VH及真类、混淆类各2个实际源参考，另外查看189和85的未标注原始VV/VH PNG。没有训练、重新提骨干、换checkpoint、重新挑参考或操作Git。

## 1. 结论与整体范围

**anchored相对full有平均正增量，但当前外部结果不支持其约束整体优于独立匹配、body_match或factorized。** 个例确实显示不同局部证据会改变结果；这与“body对应必须传递给上下文是更好的总体策略”是两回事。

下表为相同3075观测、三个seed的均值，单位为百分比，取自 `external_confirmation/metrics.csv`：

| 方法 | OA | BA |
|---|---:|---:|
| frozen full | 44.5420 | 38.3404 |
| anchored | 44.8347 | 39.1595 |
| independent | 45.3984 | 39.8973 |
| body_match | 45.3442 | 39.7609 |
| factorized | 45.2466 | 39.8388 |

anchored相对full为OA +0.2927pp、BA +0.8191pp；相对independent为OA -0.5637pp、BA -0.7378pp。不能用下面固定seed17的代表例替代这些全体比较。本集合是与当前源角色隔离的历史外部覆盖，不是新采集盲测。

另一个必要边界：所有匹配头都在冻结full之上加残差。`body_match`只是新增匹配输入仅保留body，并不是完整模型只看框内船体；full本身已经含旧完整链的上下文。layer1理论RF为35，框内池化的激活也不等于只依赖框内像素。

## 2. 重放、ID与图像范围

- 外部与源 `tokens.npz.instance_ids` 分别与各自table逐行完全一致；外部strategy的instance_ids及 `s0/body/full` 轴名称核对通过。
- `local_scores.refs` 被解释为上游source table索引，范围及八类参考class_id一致性通过，不拿外部行号替代源参考行号。
- 六个选中query的anchored、independent、body_match、factorized头，使用其保存scaler/checkpoint和实际缓存输入重放，分数最大绝对误差全部为0，top1完全一致。四者选中步分别为45/55/80/80。这里是六例重放，不冒称重新重放了3075例全部头。
- 六例真类/当前anchored最强错误类的实际参考，重新计算局部算子；两参考平均后的anchored/independent/factorized输入与保存缓存最大差 `2.384185791015625e-7`。检查容差为atol `5e-5`、rtol `1e-5`。
- 最终score NPZ没有单独ID字段；其行序还通过上述缓存索引、源码契约及六例完整头重放交叉核对，未声称从该文件独立读到了ID。
- 6图均为2×5面板：query、真类2参考、rival 2参考；每列都有VV/VH。每个面板显示完整160×160实际裁补输入，范围 `[-0.5,159.5]`，没有旧112像素显示裁切，没有resize输入或改强度。黑色padding不是海面。

精确记录见 `external_cases_receipt.json`、`case_paths.json`、`case_scores.csv`、`case_region_scores.csv`、`case_correspondences.csv` 和六份 `evidence_*.npz`。图中文字已实际查看，无标题或目标被显示裁切的问题。

## 3. 预定选例与分母

固定seed17、每个结果组取最小数值obs_id，没有挑最大改善或最醒目的图。救回/误伤方向统一为“比较方法 → anchored”。rival固定为anchored中真类以外分数最大的类；各方法自己的最强错误类也保存在 `case_scores.csv`。

| 比较 | 结果 | 组内例数/3075 | 所选obs_id |
|---|---|---:|---:|
| full → anchored | 救回 | 100/3075 | 968 |
| full → anchored | 误伤 | 71/3075 | 431 |
| full → anchored | CC | 1276/3075 | 35 |
| full → anchored | WW | 1628/3075 | 189 |
| body_match → anchored | 救回 | 67/3075 | 85 |
| body_match → anchored | 误伤 | 67/3075 | 1359 |

因此seed17相对full净+29，相对body_match净0；这不是三seed平均净收益。下面分差均为固定真类减固定rival，不是概率差。

## 4. 六例实际观察

### 968：集装箱船被救回，但不是附着约束独有

图：`case_full_rescued.png`。query来自Port Klang；真类参考26479/Santos、21323/Sydney Botany；bulk_carrier参考33678/Ningbo-Zhoushan、26615/Shanghai。

VV主体是较宽亮块，周围有细长亮响应延伸；VH主体更紧，延伸不如VV明显。两条真类参考与query均呈亮的长形/块形，但混淆类参考也有类似明亮长形，不能只凭外观断言部件类别。query的VV/VH OBB约25.29×12.14与21.84×9.62像素。

full真类分差-0.063844，anchored变为+0.230015；independent、body_match、factorized也全部救回，分别+0.123758、+0.272177、+0.009467。body_match分差还更大。

局部真类减rival的VV body/side匹配摘要分别+0.10462/+0.17345，VH分别+0.01596/+0.04590，说明现有局部输入确有真类相对正信号。但去掉T的关联项以后仍能救回，**本例不证明body–side依附是必要原因**。

### 431：bulk误伤为container，跨类局部外观相近

图：`case_full_harmed.png`。Richards Bay query；真类参考23272/Newcastle、23781/Ningbo-Zhoushan；container参考21731/Port Said、23556/Tanger Med。

query的VV背景显示纹理较亮，VH背景较暗；主体在两pol均是细长亮区。真类参考23272有相近长形，但另一真类参考方向、局部亮度分布不同；container参考同样具有连续长亮区。没有从这组图中观察到可直接命名的物理类别标志。

full为+0.038881的薄优势；anchored为-0.142888，其他三个匹配头也误伤。VV body/side局部真类分差为-0.10712/-0.12307；VH body为+0.05410，side为-0.02522，两个pol和区域并不一致。

这是**局部相似度与类别区分不一致**的实证病例，不是anchored单独的故障。当前图不能区分源特征判别性不足、参考代表性与最终头利用方式各自的因果份额。

### 35：持续判对，细槽不等于独立部件证据

图：`case_full_CC.png`。Tanger Med general_cargo；真类参考21927/Jebel Ali、12625/Houston；product_chemical_tanker参考9086/Jebel Ali、24840/Singapore。

query主体很短，8个body槽标号挤在小亮区内。VV/VH OBB长度约14.08/13.10像素，小于layer1理论RF35；body细槽的有效网格数中位数约1.96/1.94。多数参考主体更长，不能把相同槽编号当成相同分辨率的实物部件。

full/anchored分差+0.418270/+0.455964，五方法全部判对；independent为+0.527521。多数原始局部真类相对摘要并不为正，例如VV body -0.06698、VH side -0.04130，持续判对主要不能归功于一张“真类都更相似”的匹配图。需要保留已有full证据与非线性头共同作用的解释。

### 189：持续错判，同时暴露参考可用性混入相似度

图：`case_full_WW.png`。Rotterdam general_cargo；真类参考3536/Shanghai、5626/Antwerp-Bruges；fishing_vessel参考15663、16248，均来自Ningbo-Zhoushan。

实际未标注PNG中，VV/VH均整体很暗，只能见到很小的亮点/短响应；这只是显示数值观察，不是物理回波单位或噪声机制的认定。query的OBB约14–15像素长，body neff中位数1.80/2.05。真类参考5626图像较亮且VH没有有效局部token，不能把这个空比较当成“不像真类”。

full分差-1.333109；四种匹配增量都继续错，anchored降到-3.570143。存在一项可以精确定位、无需猜测的输入组成效应：

| VH body实际参考 | 匹配摘要 | 匹配质量 |
|---|---:|---:|
| 真类3536，有效 | 0.579693 | 0.937585 |
| 真类5626，无有效局部token | 0 | 0 |
| 错类15663，有效 | 0.555568 | 0.938712 |
| 错类16248，有效 | 0.567047 | 0.941368 |

固定两参考平均使真类摘要成为0.289847、质量成为0.468792，而错类平均摘要约0.561308、质量0.940040。**唯一有效的真类参考body摘要并不比两条错类差；均值中的负类间差有明确的可用参考数成分。** side/end同样存在该缺失参考置零后参与固定平均的事实。

这不证明修正平均就能救回189：VV的真类局部摘要也落后，full本来已经错，后续头同时读取相似度和质量。它说明下一步不应把此例单纯解释为body对应不可靠，更不能把所有差额当成船型外观证据。

### 85：相对body_match救回，实际是保住full已有正确结果

图：`case_body_match_rescued.png`。Ningbo-Zhoushan bulk_carrier；真类参考21013/Ningbo-Zhoushan、26615/Shanghai；general_cargo参考3536/Shanghai、7884/Jebel Ali。

未标注VV/VH中，query有近竖向细长响应和不均匀亮点序列；VH可见较长的上下延伸。query长度约35–36像素，而展示参考大多更短或更紧凑，存在实际尺度/支持差异，不能将延伸直接命名成尾迹。

full已正确，分差+0.425735；body_match误伤至-0.440818，anchored保持正确+0.164864，independent也保持正确+0.155449；factorized为-0.276609。因此本例是“相对body_match救回”，不是anchored救回了full。

VV/VH的body和side原始真类分差均负，端部则为+0.03665/+0.04190。T相对外积带来的端部类间分差增量为+0.02383/+0.01133。这显示端部摘要和关联项确实不同，但它们只是头输入，不能直接当成最终logit的端部加性贡献。独立匹配也成功，不能仅凭此例认定共享body T必要。

### 1359：body_match独自救回，更多区域没有兑现收益

图：`case_body_match_harmed.png`。Jebel Ali product_chemical_tanker；真类参考7826、9086也来自Jebel Ali；bulk参考23781/Ningbo-Zhoushan、23272/Newcastle。港相同不等于MMSI/采集相同，也不是独立泛化证明。

query与真类参考均有紧凑长亮主体，bulk参考也有非常相似的长形。VV/VH body长度约28像素，细槽neff约4；对应图中的多个局部仍可能受到同一RF的混合影响。

full本来就错，分差-0.520718。body_match救回+0.067481，anchored仍错-0.188238，independent/factorized也错。因此“相对body_match误伤”不应写成anchored破坏了full正确预测。

此例尤其说明**匹配摘要优势不等于分类优势**：anchored输入中VV body/side/end真类分差均正，分别+0.15436/+0.04992/+0.09955；VH body/end也为正，仅side -0.03370。很多T相对外积的类间增量也为正，但最终真类仍未胜出。可能涉及旧full负分差、源端学到的输入组合及外部分布变化，当前证据不能将失败全部归咎于运输几何，也不能靠换漂亮连线解释掉。

## 5. 数值分解的使用范围

`case_feature_diagnostics.csv`记录各例OBB尺寸、layer1 body neff和两参考平均匹配质量，以及 `0.5*sum(abs(T-T_outer))/sum(T)`。这些例子的对应并非数值上退化为外积，但“有非外积分量”本身不代表分类需要它。

`case_local_margins.csv`由 `case_region_scores.csv` 中每类两参考平均后，计算真类减固定rival。`score_T_minus_outer`是局部输入的精确差，不是不同已训练头之间的最终score差；本轮没有对固定头作新的输入干预。

当前factorized先对body T构造同边际外积：若 `r=T1, c=T^T1, m=sum(T)>0`，则 `T_outer=rc^T/m`。它保持body节点的行列边际与body总质量；随后才施加各区域的端部/有效性mask。**这些保持性质不保证end-to-end子集总质量相同，也不代表保留了所有区域的全部局部权重。** `case_local_margins.csv`已实际显示end mass差异，例如968的VV端部真类减rival质量，T下为+0.02850，外积下为-0.01073。

因此端部的T−外积分解同时含有关联与“质量是否落在端部子集”的差别，不能称为保持所有区域权重不变的纯配对效应。主流程报告的源端固定头T替换翻转，说明具体body对应及其端部继承共同改变了判断，**不足以排除end质量变化的解释**；它也不等于本轮外部不同已训练头之间的比较。本补充保留既有结果，不重画、不扩训练；后续按主流程安排先做source-only表示诊断，不据此追加匹配算子。

## 6. 指向的后续问题，而非新一轮结论

**优先处理参考比较的可比性，再决定是否强化附着约束。** 最直接的新证据来自189：有效参考数与相似度被固定两参考平均混合。若主流程决定继续，最小针对性问题是：在源端预定规则下，将有效参考的外观聚合与独立的可用量信息分开，能否减少这种跨类别可见性不一致，而不让大量拒配或少参考产生虚假高分？应保留当前结果，不凭外部189去选阈值或换参考，也不能简单删掉质量信息、默认归一化必然更好。这是参考证据组织问题，不是再加类别分数门控。

同时保留两个已出现但未证实根因的缺口：35/189的目标小于RF且细槽有效格少，提示当前局部描述子可能没有足够独立部位信息；431/1359显示跨类外观相近及“正的局部类间摘要未必兑现为正确分类”。若这些问题在合法源端也成立，改进应落在局部表示的尺度适配/类别判别性与参考证据使用上，而不是预设更强几何约束必然能解决。

本轮可以保留“局部匹配对full有一定增量，部分上下文信息影响个例”的结论；不能保留“body锚定的共享对应已优于普通局部比较”的整体主张。六例解释的是可见条件与具体数值路径，不代表这些条件在3075例中的频率或因果份额。所有图已实际查看；未做新训练或外部调参。
