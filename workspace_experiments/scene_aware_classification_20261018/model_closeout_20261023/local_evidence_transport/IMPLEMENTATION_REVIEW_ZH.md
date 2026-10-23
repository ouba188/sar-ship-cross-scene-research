# 局部对应读出：有界实现审阅

2026-10-23。只读审阅`extract_tokens.py`、`match_readout.py`、`train_readout.py`及其实际参考库/索引；未训练、未修改实现、未重复原图与结果分析、未Git。

读取版本：extract 22:16:16，match 22:22:00，train 22:31:27（本地文件修改时间）。审阅期间`controls_training_complete.json`已存在，9个新增对照头已有回执；未依据它们的evaluation成绩改动审阅标准。

## 结论

**未发现需要阻止当前结果分析的实现错误或实际数据泄漏。** 新增score_only及body_match的实现确实针对“只是进一步分数校准”和“只是额外框内案例证据”两个替代解释；应先比较这些已运行对照，不能仅由anchored layer1相对冻结full的BA提升认定附着邻域对应有效。

需要限定两处读出定义，见后文；它们不要求废弃现有结果或立即重训。

## 已核实的关键实现

|事项|代码/实际证据|判断|
|---|---|---|
|原尺度、C4与细槽身份|extract第85–109行：原图与同一组槽mask同步旋转，骨干各自前向，按原槽标签池化后除4累加|符合目标，没有旋转已经提取的特征图；低neff仅记录，不作>=4门控|
|槽质量与空支持|extract第24–40、103–113行：父区域软权重分配到20槽，真实footprint参与；按每视图有效质量归一，空槽输出0|没有把pad当成有效细槽；极化仍分开|
|实际ID对齐|读取tokens.instance_ids并与当前table逐行比较，完全相同|未发现跨缓存错行|
|PCA与源参考范围|match第149–162行：PCA仅fit_train有效槽；每类8艘独立MMSI的代表；实际两层各64条bank均在1205个fit_train内|符合限定范围；bank的local_index/obs_id/MMSI与table实际对应|
|源查询隔离|实际逐层检查64条bank与utility_train644、selection612、evaluation485的交集|六组比较的MMSI、pass_group交集均为0。全表虽也计算了fit_train自己的匹配，但新head不使用这些行训练/选点|
|各臂共享参考|match第163–186行：每类两条参考只按body描述预选，各臂共用refs，并沿参考轴平均|不是每臂独立挑更容易的案例，也不读取查询真类选择参考|
|运输轴与拒配|match第33–57、79–127行：query为行、support为列；null行列补足相同总质量；只汇总真实运输质量，不对拒配后剩余质量再次归一|数学定义与当前带拒配的读出相符；空支持最终被pair_score屏蔽|
|body锚定侧/端摘要|side的8槽与body轴向/横向编号一致；END及IS_END仅将两端对应到最外侧body槽；body_T用于三个摘要|没有把T转置或误用class/pol轴；end质量只计算外侧body运输到外侧的部分，不应与独立end运输的单位质量直接同义解释|
|实际三策略基底|train第68–70行取strategies[:,2]；上游保存顺序为s0/body/full|新头确实加在冻结full上，不是暗中换成body或较优旧头|
|训练及选点|train第36–49、85–107行：scaler仅utility_train；std<=1e-6列统一置零；200步AdamW .003/.01；selection exact J每5步含0，严格大于才更新|源目标及earliest tie一致。Head末层全0，初始残差严格为0；三个seed每个臂重新相同初始化|
|score_only|train第39–41行：本类中心分数1+其余排序分数7+对应margin7；同15→12→1共205参数|不是原来的仅3个分数输入；常量列处理、目标、优化、选点均与匹配臂一致|
|body_match|train第79–81行取anchored缓存，再将region轴1:的side/end两统计全部清零；正常走同一scaler|遮罩轴正确；清零后对应列恒为0，未残留side/end匹配量。名义205参数相同、有效输入更少，应如实表述|

`joint_loss`为普通CE与八类均衡CE各半，`exact_j`为精确分数实现的(OA+BA)/2。三seed仍作用于同一644/612/485样本划分，不是三套独立样本。

evaluation不进入loss、scaler、PCA、参考选择或checkpoint判定。执行顺序并非“所有头完成后才首次计算evaluation”：基底指标在训练前计算，各头保存selected后即计算其evaluation并打印（train第71–73、105–118行）。当前代码没有把这些数值反馈选点，因此不是可见的选择泄漏；记录时不要将该顺序写成全体模型封存后的首次盲评。该队列本就属于已开发的source evaluation。

## 两项需限定的实际定义

### 1. pooled不是原始区域均值余弦

位置：match第149–155、88–92行。实际先对每个细槽做`x_i=unit(PCA(f_i))`，再计算：

`pooled = sum_ij a_i b_j <x_i,y_j> = <sum_i a_i x_i, sum_j b_j y_j>`。

因此它是**归一化细槽描述的区域加权均值点积**，没有对最终区域均值再次unit，也不等于“先把原骨干按区域池化、再PCA/L2后的余弦”。逐槽归一非线性会保留/改变不同的幅度及方向一致性信息。

该定义适合比较同一细槽表示的平均聚合与匹配，当前对照无需作废；但结论应限定为“优于此描述空间内的均值聚合”，不能直接推广为胜过任意普通区域GAP。参考预选同样基于归一化槽的body均值，不应写成原始骨干GAP。

### 2. 两个新增对照能回答增量来源，但不等于所有校准或整系统框内对照

score_only是类共享、排序竞争类的特定205参数分数读出，不保留竞争类的命名身份。它与各匹配臂容量/训练相同，足以作为当前强度下的分数对照，但即使落后，也不能排除其他类别特异校准。实际回执中seed17/29的score_only均选0步、seed43选45步；前两者是selection回退，不是一个非零校准器已证明无用。

body_match保留与anchored完全相同的body运输、参考选择和两极化body摘要，只有side/end新增信息被去掉，能针对性检验附着邻域增量。**它不是整系统只看框内**：冻结full基底已有区域上下文，且layer1理论RF35会越过body池化框。body_match若复现收益，支持“额外body案例证据已足够”的解释；anchored若胜过body_match，才支持在该基底/参考库上的side/end附着摘要增量。

当前joint运输成本仅由body+side各半决定（match第119–120行），end仍是后续沿T读出的摘要；mispaired只打乱side依附（第170–184行），body/end不变。代码与当前具体设计一致，报告分别称“body+side联合”及“side依附打乱”，不要写成三个区域共同决定运输或全部邻域关系均被打乱。

## 对当前收益的判断边界

anchored layer1相对冻结full的正向BA是可保留的完整基底增量，当前审阅未发现应将其撤销的错误。其来源需用已完成的同层body_match、score_only，再结合independent/joint/mispaired区分；不能只从正向总BA命名为新对应机制已证实，也不因部分对照更好就抹去实际分类收益。

本轮不建议追加头或网格。对照的源选步与evaluation结果由主分析统一报告；原图和实际供体案例由主流程处理。本次仅新增本文，无运行中的诊断进程。
