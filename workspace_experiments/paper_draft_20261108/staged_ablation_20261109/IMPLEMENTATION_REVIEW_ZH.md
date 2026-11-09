# 分阶段实现的有限只读审查

2026-11-09。**结论：未发现需要暂停本轮训练、会使当前协议或主要实验比较不成立的实质实现错误。** 已纳入最新SE修正：carrier阶段训练两个SE，feedback阶段继续更新早层SE。此结论支持按已冻结协议继续有上限的测试和scout，不代表所有训练数值或泛化结果已验证。

审查范围为本目录 `staged.py`、`intervention_inputs.py`、`select_and_evaluate.py`，以及它们直接使用的初始化、RegionalGuidance、干预筛选接口和当前protocol。没有再次审计祖先实验，没有修改训练代码，没有运行GPU或加载完整特征缓存，只新增本文。

## 核实结果

| 关注点 | 结论与证据 |
|---|---|
| 干预仅改变指定极化 | `intervention_inputs.py:39` 按每个候选metadata.pol仅保存该路early_x/identity；第64至71行先复制同一查询两路原值，再替换指定路。几何、available和另一极化保留。读取索引start+k-1与写入一致。 |
| 干预缓存仅来自源端 | `intervention_inputs.py:20` 只枚举source_adapt。缓存终点在layer2[3].conv2之后、待训练conv3和SE hook之前，因此可跨本轮各arm/seed复用冻结ImageNet前缀。complete.json记录241组、3744个候选、original_error=0；本次只读该小型收据，未重算整缓存。 |
| 自己的教师且顺序对齐 | `staged.py:209`先配置adapt phase=-1；M1/M3/末端引导均未启用，仅改变requires_grad不改变载体预测。第218行用当前net生成教师，第221至222行核对索引顺序及原图分数。`intervention_inputs.py:74`通过当前网络评分，未借用Full或旧模型的logits/useful。 |
| 干预目标未被省略 | `staged.py:223`改用四类样本平衡权重；第260至265行调用实际adaptation_loss，包含CE、正确教师KL和原图教师margin floor。全部241条参与CE，useful仅筛选floor；系数与声明一致。 |
| 冻结范围与共同初始化 | `staged.py:104`每阶段先全部冻结，再解冻声明模块；adapt只解冻layer3[5]三个卷积且冻结head。六臂CPU构造核对共同同名同形状状态全等；无M2的global head列和bias也与Full对应切片全等。没有沿用源训练后的head值。 |
| 删除组件真正退出 | 直接依赖 `common_ablation.py:41` 至第64行删除旧M1/M3，并按arm删除新M1、matcher或整个区域M2/guide路径。no_m1的末端guide为mean路径；no_m2使用global head；no_m3不计算残差。`staged.py:41/55/63/71/82`的前向条件与这些删除一致。 |
| M3与冻结残差 | `staged.py:228`区分matcher.output的1e-4和其余matcher的1e-3，head为1e-5，再乘预定scale。第306行在M3阶段后冻结本次matcher残差。desc/mass/write是固定输入描述，不是从本轮可训练layer2.conv3之后截取的特征；后续早层训练不要求改变该冻结锚定描述。 |
| 末阶段读出缓存 | `staged.py:243`仅phase3建立缓存；第195至203行从当前训练链实际前向收集features/live2/masks/response/availability。phase3只更新guide.value/output/head，均在该缓存之后。`RegionalGuidance.forward`使用实时head和实时value/output，不缓存它们的输出，因此没有切断声明的训练路径。无M2臂仍实时训练head。 |
| 恢复与评价路径 | `staged.py:302`逐阶段恢复本轮完整state；phase1后重新计算fixed_delta。最终state中matcher仍在，评价时 `select_and_evaluate.py:39` 至第47行重新构造残差并先核对validation重放，再计算evaluation。缓存tensor不入state_dict不会导致默认残差丢失。 |
| 开发选优在评价之前 | `select_and_evaluate.py:9`只读取Full/Plain/SE的seed17两档validation指标，按BA/OA/NLL选scale，删除臂继承Full。第29行评价必须先读取selected.json。训练只保存small/validation分数；预读取评价输入和冻结描述没有用于训练目标或配置选择。没有发现评价标签参与梯度/筛选的路径。 |
| 协议表述 | protocol明确“最终存活算子分阶段重训”，没有声称历史变结构祖先链完全重放。固定32参考、逐阶段自身hard标记、统一clip5以及去掉旧模块辅助项与现代码一致；这些是新协议选择，不作为实现错误或要求回退历史的理由。 |

### SE 最新修正

最终读取的 `staged.py:118` 为 `if net.arm=='se':net.se.requires_grad_(True)`，carrier同时训练两个SE；第124行在feedback只继续解冻SE0。两处hook位于layer2[3]/layer3[5]的bn3之后、残差相加之前（第30至31、52、70行），不会被early缓存绕开。protocol第23行已同步说明，未声称全网络SE-ResNet完整复现。

CPU阶段参数核对执行时，第118行还只解冻SE1；随后这一个局部修正已按最终源码复核。这里不把修正前的SE carrier参数量冒充最终值，也不因这项明确的小修改重复六臂检查。共同初始化检查与其余阶段结论不受该修正影响。

## 本次执行的有界检查

1. 提取实际 `intervention_inputs.select` 函数，在CPU用两条查询及各自VV/VH结构/mean-control候选调用。分别模拟磁盘numpy数组和resident tensor：逐候选检查被改极化、未改极化、identity、几何、原缓存不变、原图教师分数及返回id/index顺序，全部通过。调用真实 `select_teacher` 的判定代码；网络仅为记录输入并返回确定分数的替身，因此此项证明缓存拼装/索引正确，不声称验证真实CNN干预响应。
2. 在CPU构造六个真实Network，seed17，对照同名同形状state以及global head切片；均全等。逐阶段检查requires_grad与BN冻结、删除组件是否为空，未发现越界解冻。只读ImageNet检查点及构造所需的小型head状态，未读取大特征数组。构造中旧head临时加载后由共用初始化代码reset_parameters，比较结果也支持最终未继承其值。
3. 用实际orders函数及protocol计算批次：carrier160步/3824次查询呈现；adapt96步/1446次；M3 120步/2876次；feedback180步/4314次；readout180步/4320次。adapt每轮末批1条、其他非整除尾批保留，均符合声明；不将呈现次数计为独立样本。
4. 所有成功的CPU核对结束均报告 `torch.cuda.is_initialized() == False`。没有启动训练、全缓存读取、原图批量重算或模型/源文件写入。

六臂CPU共同状态相等项数分别为Full412、Plain322、No-M1 380、No-M2 384、No-M3 382、SE322。数目表示本次参与同名同形状比较的state项，不是可训练参数量，也不是性能指标。

## 验证边界

主代理报告完整源干预前缀缓存original_error=0、自教师与Full adapt已经通过；本文独立核实了缓存小型收据和相关实现，不把主代理运行结果写成自己重新执行。没有实际执行GPU forward/backward、训练后原图重放或目标评价；这些由当前有上限的运行检查覆盖。没有据本次代码审查推断实验必然涨点。

**需主代理修复的实质问题：无。** 无新增重构、重训前置条件或历史审计要求。本文覆盖最后读取的staged.py（17020字节，修改时间2026-11-09 23:07:34）、intervention_inputs.py（4658字节）、select_and_evaluate.py（2767字节）；后续未读修改不在本结论内。
