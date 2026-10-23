# 固定源尺度asinh矩修正

2026-10-23，先锁协议再执行。此前梯度检查已证明005904在部分源设置中有极高瞬时梯度能量，但并未证明它导致整体退化。本轮是**标准稳健性修正的单一可检验对照**，不宣称asinh原创，不将数据尾部预设为有害异常。

## 唯一改变

每fold仅从source OOF训练行、各极化sigma的所有样本×8类别坐标计算s_p=median(sigma)>0。当前两极化全部valid；不按标签或样本权重加权该中位数，不逐seed或逐矩改变s。对sigma、kappa均做m'_p=s_p*asinh(m_p/s_p)，同pol共享一个尺度。outer验证/测试和源OOF都使用该源尺度，不再估计。该单调奇函数保留符号、顺序以及|kappa'|<=sigma'，不做clip、不改mu/B/valid，也不删除任何样本。sigma/kappa经变换后应称“压缩矩”，不再冒称原空间标准差/有符号矩的精确空间积分。

只训练local209与additive266，6fold×seed17/29/43=36头。A、T、PCA、W、划分和原90头全部冻结。复用原descriptor定义，对压缩后的source OOF输入重新fit原样scaler；mu和B原值不变。网络容量、随机初始化、AdamW lr .003/wd .01、200steps、jointCE、每5步validation J选点、epoch0及严格更优才更新全部不变。单一中位数尺度公式，无网格，无测试调参。

全部36个源选点及checkpoint冻结写盘后才统一评价test。输出predictions.csv沿用原契约，arm仅local/additive；原始头仅从其既有结果和模型读取，不重训。分别报告对同一B和对原未压缩同臂的比较，不择优替换预先确定的候选。

## 验收与机制评价

检查尺度只源产生且跨seed相同、有限正值、单调/符号/幅度不等式、零输出=B、选点checkpoint源validation J重放、全部test身份覆盖与原B一致。记录scales、selections、curves和耗时。

使用真实单样本jointCE系数0.5/N+0.5/(8n_class)，在初始化和已选状态对全部源行计算所有参数梯度平方范数。复用此前原头梯度表作配对，输出005904、最大个体、top5/top1%能量、有效贡献样本数、类别能量及有符号合梯度投影。先看整体分配是否改变，再讨论尾部修正，而不以矩方差下降作为机制成功。

平方范数占比不是AdamW更新占比或完整训练轨迹；selected比较的epoch可不同。原图分析由另行支线负责，本实现只交付训练、数值机制对照及边界说明。不修改父脚本globals、不覆盖原结果、不Git、不新增臂。
