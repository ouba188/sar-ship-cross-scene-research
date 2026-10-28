# Shallow M2 数据与原图依据

2026-10-28。本说明在本轮完整预测产生前写成。已阅读当前发布包 `VERIFICATION_ZH.md`、原生输入实现、当前实验 `PROTOCOL_ZH.md` 及长期工作流中的模块组合规则。本支线只做CPU数据分析和结果分析脚本，不训练、不改 `run.py`、不操作Git。

## 实际覆盖

统计直接来自 `E:/临时会话/sar_m123_current_20261027/manifests/` 下的 `source_adapt.json`、`development.json`、`references.json`。逐采集ID计数和组间交集保存于 `data_summary.json`，可用 `python -B analyze.py --data-only` 复算。

| 清单 | 观测数 | 不同MMSI | 不同pass_group | 港口数 | VV+VH | 仅VV |
|---|---:|---:|---:|---:|---:|---:|
| source_adapt | 136 | 128 | 20 | 10 | 128 | 8 |
| development | 160 | 158 | 3 | 3 | 145 | 15 |
| references | 16 | 16 | 13 | 9 | 15 | 1 |

这里“采集”是manifest的pass_group，不把同一采集中的多个chip视为独立采集证据。没有仅VH样本。表中极化可用性按manifest非空视图统计，不能将缺失VH当作实际观测到的全黑VH。

| ID | 类别 | 源观测 / 船只 | 开发观测 / 船只 | 参考观测 / 船只 |
|---|---|---:|---:|---:|
| 0 | bulk_carrier，散货船 | 54 / 52 | 25 / 24 | 2 / 2 |
| 1 | fishing，渔船 | 4 / 4 | 0 / 0 | 2 / 2 |
| 2 | general_cargo，杂货船 | 26 / 25 | 12 / 12 | 2 / 2 |
| 3 | chemical_tanker，化学品船 | 28 / 26 | 70 / 70 | 2 / 2 |
| 4 | container，集装箱船 | 6 / 5 | 22 / 22 | 2 / 2 |
| 5 | crude_oil_tanker，原油轮 | 12 / 11 | 16 / 15 | 2 / 2 |
| 6 | tug，拖轮 | 3 / 2 | 7 / 7 | 2 / 2 |
| 7 | offshore，海工船 | 3 / 3 | 8 / 8 | 2 / 2 |

开发集BA只平均出现的7类；8类输出与参考库均保留渔船类。源端散货船54/136，开发端化学品船70/160，因此OA和BA变化可能对应不同类别权衡。逐类统计必须与总指标一起阅读。

源适配港口：Busan 21、Callao 4、Fujairah 17、Houston 8、Los Angeles 25、Newcastle 7、Port Said 36、Richards Bay 1、Santos 16、Sydney Botany 1。

开发集的港口与采集组一一对应：Jebel Ali 69（S1A_058688_074483）、Singapore 63（S1A_060268_077D6F）、Port Klang 28（S1A_060692_078D67）。在这160条中，港口差异与采集条件无法单独辨识，不能将分港相关现象直接归因于港口或某种物理条件。

参考港口：Houston 1、Jebel Ali 7、New York 1、Ningbo-Zhoushan 2、Port Klang 1、Port Said 1、Shanghai 1、Singapore 1、Tanger Med 1。

三组两两比较的ID、MMSI、pass_group交集均为空。源适配与开发港口也无交集；参考与源适配共享Houston/Port Said，与开发共享全部3个开发港口。这是固定参考组成的事实，不能把本轮描述为参考也来自全新港口的完全未知港口确认。

136条是小范围源适配，不是重跑1205条基础拟合；160条是长期复用开发集，不能作为独立确认。128或158艘船也不能消除采集共享带来的关联。训练seed17/29/43按当前协议是训练顺序等随机条件的重复，不是三次独立数据划分。

## 实际固定16条参考

| 类别 | 参考ID |
|---|---|
| bulk_carrier | 34478、31512 |
| fishing | 14814、16401 |
| general_cargo | 6913、22755 |
| chemical_tanker | 24527、7790 |
| container | 36308、23719 |
| crude_oil_tanker | 29922、2077 |
| tug | 12041、7429 |
| offshore | 13000、13276 |

22755无VH，其余15条双极化。查询与参考无同船重叠，仍按现有ID/MMSI规则记录可用参考，不另抽样。

已生成 [VV参考面板](data_preview/reference_bank_vv.png) 与 [VH参考面板](data_preview/reference_bank_vh.png)。每格分别显示原chip与原实现生成的160像素输入；缺失极化明确标记。此处记录面板已生成，不把它自动等同于训练后主代理已经逐张复核。

## 已实际查看的2条源和2条开发原图

选例在读取本轮预测前固定：源294用于对照不同背景和极化画幅；源4256及开发104/3951复用工程验证中的已知原生输入案例。两条开发样本同为原油轮、同港同采集，用于有限范围内对照形态；它们不代表其他类别或港口。选例不使用本轮涨跌信息，无随机抽样seed。

本支线通过图像查看工具逐一打开下面4条的VV与VH，共8张源PNG。完整路径、MMSI、pass_group、原尺寸、crop/pad位置及保留像素数见 `data_preview/native_sources.json`。这里只描述可见现象，不为响应强行命名物理来源。

| split / ID | 身份与采集 | 原尺寸 VV / VH | 本次直接可见事实 |
|---|---|---|---|
| source / 294 | bulk_carrier；MMSI 209796000；Newcastle；S1A_059019_0751F3 | 80×80 / 96×96 | 两极化均有偏竖直的斜向细长亮区；VV周围颗粒状亮暗纹理较明显，VH背景较暗。VV/VH原画幅不同，不能按同一像素坐标强作跨极化细节对应。 |
| source / 4256 | crude_oil_tanker；MMSI 255806059；Los Angeles；S1A_057536_071629 | 72×72 / 72×72 | 两极化中央附近都有从左下向右上延伸的短亮带；VV背景更亮且纹理明显，VH背景更暗。不能由此推出外部纹理有益或有害。 |
| development / 104 | crude_oil_tanker；MMSI 205517000；Jebel Ali；S1A_058688_074483 | 80×80 / 80×80 | VV中央近竖直亮区周围有横向及纵向延伸、重复条纹状响应；VH亮区较集中，周围延伸较弱。不能仅凭外观称其为已确认的旁瓣或其他具体物理机制。 |
| development / 3951 | crude_oil_tanker；MMSI 249611000；Jebel Ali；S1A_058688_074483 | 80×80 / 80×80 | VV/VH都呈左上向右下斜亮带，可见若干亮点延伸；主体与周围背景的对比和分布与104不同，但这种差异本身不是类别正确性解释。 |

相应面板：

- 294：[VV](data_preview/source_adapt_294_vv.png)、[VH](data_preview/source_adapt_294_vh.png)。
- 4256：[VV](data_preview/source_adapt_4256_vv.png)、[VH](data_preview/source_adapt_4256_vh.png)。
- 104：[VV](data_preview/development_104_vv.png)、[VH](data_preview/development_104_vh.png)。
- 3951：[VV](data_preview/development_3951_vv.png)、[VH](data_preview/development_3951_vh.png)。

既有 `VERIFICATION_ZH.md` 记录104在工程迁移前后持续错、3951持续对；这是历史迁移比较的事实，不是本轮浅层M2的预测或翻转结果。其原生重放容差及相同预测也不证明本轮新模块有效。

## 原始尺度与模型输入

这里“原图”指manifest中的原始8-bit UTM显示chip，不是校准后的原始SAR复数数据。显示保留原0–255值，不做单图归一化、增强对比或缩放。

面板复用发布包 `sar_m123/input.py::prepare_view`。模型输入保持160×160画布，目标居中裁剪或补零，input_scale=1。上述8图均整幅保留，无裁剪：294 VV四边补40、VH补32；4256补44；104与3951补40。OBB目标像素保留量与原量相同。原图和160画布在面板中均按1像素对应1输出像素排列；160画布黑边属于补零，不是观测到的背景。

## Shallow细网格区域读取假设

已有模型在layer3.5内部执行M2池化回写，较浅特征也参与后续区域读出；本轮检验在layer1或layer2阶段输出先按区域池化并回写后，后续卷积能否利用这些调整。不是假设浅层信息原来完全没有进入分类器。

当前协议中layer1为40×40、256通道，layer2为20×20、512通道；原深层路径为10×10。原生chip只占160画布的部分区域，不能将所有网格都当有效观测。294的两极化画幅不同、104的集中亮区及外延响应、4256的背景差异提示：在区域聚合前保留更细位置划分是可检验的读取途径。但这些原图没有直接证明深层聚合已错误混合证据，更不证明细网格必能涨点。

合理收益路径是：较细网格上的框内/轴端/侧向软区域占比与响应池化，可能改变信息回写的位置及后续卷积所见特征；分类收益需由实际配对结果检验。区域名是几何组织，掩膜不是纯船体分割，外侧响应也不自动等于背景、尾迹或特定散射来源。

对应四组为baseline、layer1、layer2、both，同一教师起点、固定16refs、相同136源样本/三seed/训练步数/末轮规则。baseline也获得相同追加训练机会，主结果不能只与未训练initial相比。both比单层多参数与计算，不能解释为只改变位置的等容量对照。

接口上需由主实验确认：细网格掩膜与特征对齐，有效区域/缺失极化处理一致，冻结后续卷积仍传递输入梯度，query和实际refs共享当前新增模块，M3比较锚点按协议保持。已有diagnostics提供结构自检记录，不等于任务收益。分析脚本原样收录该记录，不自动将任意数值字段概括成“全部通过”。

## 训练后分析约定

`python -B analyze.py` 读取同目录真实 `predictions.npz`、`runs.json`、`diagnostics.json`；输入必须与开发manifest的160个ID及标签准确对应。scores/initial按logits处理，使用稳定log-softmax计算NLL和8类平方和Brier。各arm逐seed与baseline同seed配对，再汇报三seed均值及样本标准差，不先集成概率。

`classes.csv`保留n=0渔船行，召回留空；`summary.json`另保存分港、分采集、双极化/仅VV分层和原运行曲线。小样本分层用于定位候选解释，不作独立因果证明。

CASES仅从实际seed17选择每arm的rescued/harmed/CC/WW，每组最多一条，按固定竞争类分差变化绝对值最大、再按ID排序；空组明确为0，不换seed补齐。组别比较相同seed的已训练baseline。参考竞争类固定为 `initial` 原基线最强非真值类：原基线错误时就是原错误类别，原基线正确时是最强错误竞争者。记录新预测但不据它替换参考类；WW即使换成另一错误类别也保持原竞争类，便于比较同一条证据轴。

每个实际案例同时提供query及实际同类/固定竞争类参考的原生VV/VH面板，HTML末尾提供完整实际16refs。选例不等于读图完成；主代理须补充可见事实、与分差/损失一致或矛盾之处，以及仍无法解释的部分。无翻转时仍检查CC/WW的证据变化，不以代表案例推断总体比例。

当前没有从本轮完整预测得出的收益、退化或物理机制结论。下一步是主代理完成训练后运行脚本、查看实际案例并作最终判断；本支线不追加训练或调参。
