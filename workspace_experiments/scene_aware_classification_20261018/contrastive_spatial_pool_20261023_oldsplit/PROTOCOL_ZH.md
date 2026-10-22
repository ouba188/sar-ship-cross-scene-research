# 旧 split 固定 layer3 std 复验

冻结协议：旧 context_expansion_20261022/fold{0..5}_inputs.npz，5695 条原队列，按已有 MMSI/pass 划分，不采用新 partition，也不采用 source-selected stem 混合读出。

重用 deep_features.npz 的原尺度 R50 layer3 GAP/C4 特征及 run_strong.View/fit_head/risk。每折仅源 train 拟合 StandardScaler/PCA128/postscale 与加权三类 Ridge，alpha=[1,10,100] 按源 validation balanced NLL 选择，logit scale=3。必须先核对历史 fixed-layer3 alpha、逐例 top1 和分数，复现 OA81.51009657594382/BA80.12973869349547，测试 A 继续使用历史 CSV 的精确值。重建仅为取得原通道 W，不能静默替换基线。

读取已有 native_targets VV/VH PNG，不整景解码、不重新检测、不改像元或标签。R50 checkpoint、ImageNet 归一化及 C4 与原提取相同，无额外 resize。每 pol 每 rotation，v=W(F-Fmean)，r=v-mean_class(v)，delta=population std_position(r)，先双极化求和再 C4 平均；不将 std 命名物理方差。只有 contrast_std 一臂，lambda=[0,.1,.25,.5,1]，含0，仅源 validation balanced NLL 选择。无新注意力训练，不追加控件或扫描。

验证覆盖：历史基线逐例重放；原通道 affine 重建；fresh GAP 与原缓存；非空有限/完整覆盖；保存 lambda 候选及逐例分数。结果给出逐类/采集、pass/MMSI 固定预测配对 bootstrap，非训练或选择不确定性。原始 VV/VH 对 rescued/harmed/CC/WW 各查看一例；首 ID 确定性抽样，不按外观选择。案例不代表总体比例，空间图粗且 RF 混合，不作精确 body/bg 结论。

仅本目录输出。父实现只读复用指标和选择函数；不改父文件、不 Git、不安装依赖。负结果同样保留。
