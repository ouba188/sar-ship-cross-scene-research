# Free 读出的同容量信息替换协议

2026-10-23，训练前冻结。仅本目录写入；不修改父文件、不Git。只训练 free_aggregate / free_constant 两臂，各3fold×seed17/29/43，共18轨迹。旧 standard_controls/free_shared 的真实信息路仅读取冻结模型与结果，不重训、不选赢家、不扩网格。

## 信息与结构

完全复用 standard_controls.Head('free_shared')：11→16 GELU→1，逐类共享、输出减类均值，209参数；无tanh上界、无κ乘法载体。相同seed的初始参数完全相同，最后一层为0，epoch0回到同一固定B=C(A)。逐图B和有效极化flags始终保留。

- free_aggregate：对所有矩输入调用 run_polarization.organized(...,'aggregate_repeated')。有效路等分原VV/VH矩之和，无效路0；保留逐图聚合σ/κ，去掉原始路别分配。描述符中的绝对矩及类中心化矩均由替换后值生成。
- free_constant：从当前fold源OOF train矩，分别只取该pol valid行，算固定[pol,class,moment]=[2,8,2]均值。训练、validation、test均用该均值，再按各样本valid遮罩。不是从outer train、validation或test求均值，不用标签分组。B仍逐图，故输出不是全局常数；类固定矩参照和flags也仍可被利用。209名义参数相同，但有效输入变化如实说明。

两臂各自只在替换后的source OOF train描述符上拟合标准化，沿用父inputs的轴和尺度下限。源A及局部矩为匹配inner-held OOF；C为既有冻结校准头，非二级OOF，不能称整流水线OOF。原W、划分、标签、native160crop/pad、C4、footprint与B不变，无新提取、无Reader拟合。

## 优化与选点

每头200步，AdamW lr=.003、weight_decay=.01；jointCE=.5普通CE+.5八类CE均值。每5步source validation J=.5OA+.5BA，含epoch0，精确J严格增加才更新，平局保留最早，不加NLL择优。CPU、float32训练，与旧free保持同初始化方法。保存全部18个选点后才统一test；最终float64 B+float32残差的validation J必须与所选J一致。重放保存头和原B，不因数值误差改模型或放宽选点。

## 输出、图像与结论范围

保存模型、常量及有效行数、全部选点曲线、时间、预测、逐折/类/采集/sourceval/OA/BA/NLL/Brier、零头和重放检查。test每seed同4744唯一对象。sourceval同时报告折暴露和唯一对象，不能把915折暴露当915独立样本。

各seed对B及冻结真实free逐例配对，pass与MMSI两种2000次块bootstrap均报告，不择优；区间条件于已保存模型，不覆盖训练/选点不确定性，不把3seed当独立对象。分析真实free减两个信息替换，分清逐图矩与路别分配；控制表现好不等于全部局部信息无用，真实路表现好也不等于机制或一般必要性已证明。

固定seed17，每新臂rescued/harmed/CC/WW各取最小instance_id，实际查看原VV/VH和实际160窗口、记录旧/新竞争类与分差。若某结果缺失如实说明。constant图只展示输入来源与预测，不伪装逐图κ贡献图，不从外观命名物理机制。完成18头及本目录FINDINGS即停止。
