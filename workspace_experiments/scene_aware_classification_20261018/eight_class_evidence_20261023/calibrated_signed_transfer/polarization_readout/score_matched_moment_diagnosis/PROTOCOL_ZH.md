# 分数匹配的整组矩交换：冻结诊断协议

2026-10-23，运行前锁定。只写本目录，无训练、无新头、无epoch/阈值选择、无Git，不改父文件。唯一对象为full_model_integration/free_real的3fold×seed17/29/43共9个已选209参数头；B指该实验原旧A+D，不是新支校准B。完全冻结头、B、descriptor scaler和原矩缓存。

实施偏差说明：下文“不先做逐行类中心化”遗漏了主代理原先约定，后由主代理设计一致性检查发现；不是用户改变标准。保留下文作为uncentered历史实施记录，正确centered协议与结果在centered_correction/，不将原结果冒充设计一致的主结果。

## 匹配规则

1. 用父load_old恢复各fold/seed源OOF train B，按样本轴计算8类各自mean/std（population std，下限1e-8）。距离为标准化完整8维B的欧氏距离，不先做逐行类中心化；标准化均值抵消于距离，但一并保存。该匹配尺度不替换冻结head的descriptor scaler。
2. validation与test各自独立；仅同fold/seed、相同B argmax、相同validVV/VH模式，且不同MMSI的对象允许相连。匹配函数不接收GT。每组对象按instance_id排序，列出所有合法无向边，按距离、左ID、右ID依次排序；从最短边起贪心接受两端均未使用的边，每个对象至多配一次。只此规则，无随机扫描、无距离阈值、无全局置乱。
3. 每对互换整个[2pol,8class,2moment]矩块，不拆σ/κ、类轴或VV/VH。B和flags不动，所有矩相关descriptor重算，scaler与head不动。未匹配对象保持原矩。记录单例条件组、没有不同MMSI候选、合法候选已被前序贪心占用三类原因；不为提高覆盖率放宽条件。
4. 保存逐例donor/距离/源尺度、组大小、候选数、匹配比例、对象/MMSI/类别/过境实际构成、未匹配原因。GT仅在配对冻结后用于同类率及评价。核对互换双向一致、一一使用、有效模式/预测类相同、MMSI不同、未匹配原样。

## 重放与评价

先用相同role batch重放original，核对当前full_model_integration/predictions.csv的free_real逐分数（atol1e-7、rtol0）及全部top1；失败则定位，不放宽误差/改模型。B亦核对。保存实际最大误差。

三种固定分数为B、original本图矩、swapped借用矩；主要信息效用方向定义original−swapped，正值表示本图矩更好。分role/seed保存OA/BA/J/NLL/balNLL/Brier及逐fold/类/pass/MMSI构成；全量含未匹配为主，matched-only另列以显示稀释程度。sourceval按原折暴露报告，并列唯一对象数，不能将跨fold重复曝光当独立样本。

original−B、swapped−B、original−swapped均按pass与MMSI做固定预测配对2000次bootstrap（沿用20261023种子，不进行匹配种子搜索），sourceval/test分开。区间条件于已选头和本次固定配对，未重做匹配/模型选择；donor互换可能跨block，区间不作为完整交换机制不确定性的因果检验。

## 原图与解释

固定seed17，在test已匹配对象中按instance_id取四种结果首例。这里rescued/harmed/CC/WW明确比较original→swapped（不是B→original），避免把交换影响误写成原模型增益。没有某种翻转则如实记录，不换seed补正例。

每例同时查看query及其实际donor的VV/VH；4个query+4个donor角色视图，共4个左右配对面板，每面板8个子图（双方各VV/VH native与160crop/pad）。显示双方GT、B预测、original预测、swapped预测、原/替换真类margin、距离、有效标记及原尺寸。标签只用于事后说明，绝不用来挑donor；无效路显示为不使用。case CSV同时保存双方原图路径和预测。

这是局部条件化信息干预，不是部署模型、不是像素因果遮挡，也不保证整组交换完全同分布。sourceval已参与原头选点，不能称独立确认；相同预测类/近分数也不等于相同类别/场景。正负结果均保留，完成这一规则的诊断、原图核查及FINDINGS后停止。
