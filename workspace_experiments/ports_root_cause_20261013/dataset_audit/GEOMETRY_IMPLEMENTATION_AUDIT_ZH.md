# SLC 几何实现独立审阅

2026-10-13。**未发现阻断性实现错误；所要求的轴向、等面积控制、支持集接入和指标计算均符合实际代码。** 已独立核对 fold 2 的 R/T expert，全查询、SLC查询及产品配对；不把这个范围扩写为两折所有臂的独立验证。本分支未改主代码、未拟合、未编码、未用GPU。

## 代码与控制

- [slc_geometry.py:13](E:/临时会话/ports_root_cause_20261013/slc_geometry.py:13)：图像数组为H/W，PIL尺寸为W/H；range臂压缩宽度X，sham压缩高度Y，符合ReadMe的X=range/Y=azimuth。`range_spacing/sin(incidence)/azimuth_spacing` 与已恢复的slant-range网格及局部平面近似一致，角度正确转换为弧度。没有猜用2.3/14.1或2.3/17.4。
- [slc_geometry.py:18](E:/临时会话/ports_root_cause_20261013/slc_geometry.py:18)：验证正方形输入后，两臂使用同一个 `m=max(1,round(w*ratio))`，实际内容面积分别m×h和w×m，因h=w严格相同。双线性插值相同，按floor居中放置；奇数余量可能左右/上下差1像素，但两臂规则对应，不构成额外面积偏差。
- [slc_geometry.py:22](E:/临时会话/ports_root_cause_20261013/slc_geometry.py:22)：补边取每极化自身四边灰度中位数；两臂同极化的补边值相同，不要求VV与VH值相同。实读999条controls，两臂sample_id、整数尺寸、面积、VV/VH fill与ratio全部相同。另用不对称人工图直接运行实际纯变换函数，轴向与面积检查通过。
- [slc_geometry.py:35](E:/临时会话/ports_root_cause_20261013/slc_geometry.py:35)：仅替换999条SLC，GRDH沿用原特征；代码对GRDH三组特征执行逐位一致检查。`data_for()` 返回整个替换后目标矩阵，非只替换query。
- [run_factorial.py:44](E:/临时会话/ports_root_cause_20261013/run_factorial.py:44)：从替换后的矩阵取原支持ID并重新构建支持核、先验及系数，再预测所有目标。因此支持和query中的SLC均改变；只读已有原source模型，**R/T/DD无需mapper**。所谓“无fits”应写为无新源模型/神经拟合，仍有必要的支持闭式重解。

## GRDH变化的含义

[slc_geometry.py:139](E:/临时会话/ports_root_cause_20261013/slc_geometry.py:139)只要求GRDH的prior概率不变，正确；并未错误要求expert/fusion不变。独立核对fold 2的所有臂，GRDH prior逐位相同。

五次原支持分别含16/19/19/13/19条SLC。它们的变换会改变支持核、残差系数和对GRDH的交叉核。实测fold 2、R expert的range臂对GRDH query的最大单概率改变为0.31717。这是预期任务耦合，不是误改GRDH图像的证据。因此全OSS2收益及配对中的GRDH变化不能都解释为“SLC query外观校正的直接收益”；它们也包含改变支持表示的效果。

## 分母与独立数值核对

[slc_geometry.py:124](E:/临时会话/ports_root_cause_20261013/slc_geometry.py:124)先从234个同时间一一配对中保留233个同标签对，再逐draw要求两端均为query；双端clean另作固定子集。GRDH/SLC两侧同一标签向量，配对内部BA分母一致。`metrics()` 的BA与宏Brier均只平均当前真值实际出现的类，八维概率仍保留，不把未出现类偷偷当成正确或删除预测类别。

| draw | 同标签且两端query | 实际类数 | 两端clean | clean类数 |
|---|---:|---:|---:|---:|
|0|227|8|156|8|
|1|223|8|151|8|
|2|224|8|154|8|
|3|222|8|153|8|
|4|223|7|152|7|

最后一次没有tug类别，因此平均类数7.8；不是固定八类BA。`geometry_pairs.csv` 本身记录pairs但不显式记录类数；本审计补出 `geometry_pair_denominators_fold2.csv` 的逐类计数，报告时应一起使用。这是呈现边界，不影响已计算值的正确性。

独立脚本不导入模型runner或原metrics函数；用sklearn OA/BA及等价Brier表达式 `sum(p²)−2p_true+1` 从冻结概率重算。对照范围：fold 2、五draw、R/T expert、original/query_SLC两范围、三种输入与三种差值；另核对paired_all/paired_both_clean两种配对中的GRDH/SLC OA、BA、宏Brier、gap及样本数。**840项标量对照全部通过，最大绝对误差1.0658e−14。** sklearn对七类真值中出现第八类预测的提示属于预期范围，BA仍按真值出现类计算；未调整标签或掩码。

汇总是各fold/draw指标等权平均，不是合并所有pair再计算；两折共享目标、五次支持重复，也不是十次独立确认。全查询/SLC查询、配对、clean配对不能互相替代。已验证fold 2底层表，不额外声称完整两折汇总均被本分支重算。

## 物理结论边界

早期“OSS2缺spacing”结论已撤回：元数据实际使用完整产品名.xml。全部11个入选SLC产品的精确间距已恢复，999条raw/cal/uint8尺寸匹配XML，22个幅度/显示像素探针支持原网格单调显示映射。本次审阅使用这些既有证据，没有再扫描来源文件。

range臂优于原始和等面积axis-sham，可支持**与元数据所指轴向相符的采样/显示几何处理对当前任务有效**。它仍同时改变轴向平滑、插值、目标形状、上下文与补边分布；sham控制总面积/补边值和操作量，不会使两个轴的频率损失相同。中心incidence的局部平面近似不是完整地理校正，不能由性能改善唯一确定物理根因，或把差值全部归于某一支路新增信息。此边界不否定已经测得的有限范围收益。

## 交付

`verify_geometry_implementation.py`、`GEOMETRY_IMPLEMENTATION_VERIFIED.json`、`geometry_independent_numeric_checks.csv`、`geometry_pair_denominators_fold2.csv`、`geometry_support_coupling_fold2.csv` 均在本审计目录。核对进程已退出；无后台任务、无新增实验。
