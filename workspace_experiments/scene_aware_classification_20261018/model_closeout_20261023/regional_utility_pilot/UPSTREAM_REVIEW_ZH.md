# Upstream 窄接口审阅

日期：2026-10-23。只读检查 `upstream/pipeline.py`、`extract_source.py`，并定向对照旧attention/区域输入接口；未运行提取器、真实训练或修改上游文件，无 Git 操作。

## 结论与版本

**在所读版本中，未发现将utility_train/selection/evaluation标签用于上游拟合、错误的W通道拆分或attention列错位。** 本结论是接口审阅通过，不等于当前动态提取已完成GAP重放；实际GPU提取与数值修复继续由Plato负责。

- `pipeline.py`：18851字节，最后修改时间 `2026-10-23 12:34:54 UTC`（本地20:34:54）。
- `extract_source.py`：初读12638字节、`12:40:10 UTC`；审阅期间更新为13473字节、`12:42:30 UTC`。对更新版只定向复读area权重、响应和闭合相关代码，没有重复全量审计。更新版使用160→10的CPU area bins，再传GPU。
- 小型CPU数值检查只使用已保存reader参数及合成张量，不拟合任何模型。不存在本审阅启动的训练/提取进程。

## 1. 标签、标准化和选点边界

`pipeline.load()` 从 `indices.npz` 读取fit_train/fit_validation，核对两者在table中均为strategy_fit，并检查二者MMSI及pass不交叉。后续拟合使用这些本地位置索引；保存时转换为table的原row_index，没有把两种索引混用。

逐级读取的实际路径如下：

|阶段|拟合统计/监督|选点|
|---|---|---|
|axis RidgeClassifier|h的mean/std及类-身份权重仅fit_train|alpha仅fit_validation BA/Brier|
|five-point residual|keep_refs的mean/std、MSE标签和权重仅fit_train|fit_validation BA/Brier|
|neighborhood residual|每pol只用fit_train且usable的样本拟合centered/level scaler；MSE标签仅fit_train|fit_validation BA/Brier|
|score calibration T|中心化U的scaler与jointCE仅fit_train|fit_validation精确J，含0且最早平局|
|layer1/layer3 reader|StandardScaler/PCA/postscale/Ridge及类-身份权重仅fit_train|alpha1/10/100仅fit_validation balanced NLL|
|attention、body/full|descriptor scaler和jointCE仅fit_train|fit_validation精确J，含0且最早平局|

`extract_source` 的scale/probe模式只取fit_train；温度是该范围有效极化的C4均值sigma中位数。summaries模式对所有角色做冻结前向，但不使用它们的类别拟合reader或温度。加载table含有标签列、对全部样本做冻结变换本身不是标签泄漏。

**必须保留的比较范围：** strategy_fit内部多级头使用前一级训练内预测，不是整栈嵌套OOF；上游也反复用自己的fit_validation选各级头。这些均被限制在strategy_fit角色，不能据此说router三种留出角色已被训练暴露，但也不能称完整链每一级训练输入都是OOF。新tau来自fit_train的同reader响应，旧attention的tau来自源OOF；这是当前重建协议已记录的变化。此pilot检验新源角色策略池上的router，不是对历史full9分数的精确重放或历史目标效果复验。

本次不重复核查四角色生成算法/全部身份清单；上游外部角色隔离仍以负责该部分的审查和记录为准。

## 2. Reader原通道与空间响应

`fit_reader` 显式映射为

`W = 3 * ((coef / post_scale) @ PCA_components) / input_scale`。

所有StandardScaler/PCA/postscale中心偏移进入单独intercept，代码通过零输入的完整变换计算它。此W可以直接作用于未标准化的原CNN通道；不能再对局部特征重复应用全局PCA或标准化。`extract_source` 当前没有这样重复变换。

已读取当前两个保存reader，在不重新拟合的情况下核对：

|reader|W形状|W对保存变换重建误差|随机h完整变换vs显式线性输出最大误差|fit/validation元数据|
|---|---|---:|---:|---|
|layer1|8×512|0|5.685e-13|均与本次tr/va原row_index完全一致|
|layer3|8×2048|0|7.816e-14|均与本次tr/va原row_index完全一致|

W按 `reshape(8,2,dim).transpose(1,0,2)` 拆成pol×class×channel，layer1 dim256、layer3 dim1024，符合旧缓存VV后VH的通道拼接。取当前batch的pol后为 `[B,8,D]`，与 `[B,D,U]` 经einsum得到 `[B,8,U]`；合成两pol张量输出 `(2,8,25)`，没有多出区域轴或把class当batch。

局部mu本身不加intercept，和旧接口一致；两pol全局mu求和后再加一次intercept才对应reader分数。代码保留 `sum_pol(mu)+intercept` 对 `GAP@W.T+intercept` 闭合检查。由于后续r去空间均值、去类公共项，不能把遗漏intercept误判为局部响应缺项。

## 3. C4、双向attention与15列

旧接口核对范围是 `matched_attention_readout/extract_attention.py`、`joint_distribution_readout/run_joint.py` 的公式及列定义，以及旧OBB `extract_regions.py` 的区域摘要公式，没有重查完整历史训练。

- layer3先在真实footprint内做响应空间去均值，再去类公共响应，最后再消除浮点残余空间均值；正负softmax分别作用于 `log(weight)+r/tau` 和 `log(weight)-r/tau`。保存 `h=(qplus-qminus)/2`、`b=(qplus+qminus)/2`，不是误把旧sigma/kappa继续喂给attention。
- 每pol每个C4分别计算mu/h/b，再除4累加；非线性attention不是先平均四张图再算。两个pol保留为独立descriptor槽，不提前相加。invalid pol跳过，保存值为0。
- layer1区域按自身body/end/side权重计算mu与r，公式与旧OBB摘要相同；min C4 neff≥4的flag控制摘要有效性。body只清新增end/side原始摘要，保留实际quality，与旧control语义一致。
- 合成有效支持张量逐公式核对旧layer3双向attention，新旧packed(mu,h,b)最大误差 `3.331e-16`，没有运行骨干。

`attention_raw` 列为：

```text
0: centered B
1,2: VV mu, centered mu
3,4: VV h, centered h
5,6: VV b, centered b
7,8: VH mu, centered mu
9,10: VH h, centered h
11,12: VH b, centered b
13,14: VV/VH valid
```

`AttentionHead.mean` 取 `[0,1,2,7,8,13,14]`，local取 `[0,3,4,5,6,9,10,11,12,13,14]`，与旧266参数additive attention一致，末端两支相加后去classmean。合成descriptor为 `(2,8,15)`，VV mu、VV h、VH b实际列均逐值相同。`scaled` 对该类共享输入只在tr的sample/class两轴拟合统计，未把validation或router角色纳入。

## 4. 提取尚在修正的边界

更新版layer1使用旧stride4/origin0的精确Voronoi权重；layer3在160画布对应10×10特征图上，用16×16原像元area均值实现旧footprint池化。旋转时mask和图像一起旋转。这里确认的是索引/形状/公式，**不冒充已经验证当前GPU原窗重提结果**。

最终能否用于本次router，仍以Plato跑通的原GAP重放、两pol均值线性闭合及实际有效mask记录为准；代码保留这些会阻止错位缓存继续使用的检查。没有据本次审阅要求额外重提、放宽误差阈值或接手当前提取调试。

最终判断：本次范围内没有需要主流程暂停router设计、修改标签、重划分或更换attention列的实际问题。仍须将当前结果称为独立source角色下重建上游的pilot，不能借接口一致性宣称与历史完整模型的分数/训练组织完全相同。
