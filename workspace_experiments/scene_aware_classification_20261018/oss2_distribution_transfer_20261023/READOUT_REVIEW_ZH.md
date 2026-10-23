# OSS2读出实现限定审阅

2026-10-23。**未发现需要修正或阻塞当前运行的实质逻辑错误。** 当前extract_moments.py、train_heads.py与MODEL_PROTOCOL_ZH及fit_readers输出相符。本次只读源码、缓存及现有提取回执，执行了CPU无梯度接口检查；没有训练、GPU提取、修改主脚本或Git操作。

## 已核对的数据流

1. **类别与极化轴正确。** extract_features.load_batch按每样本VV、VH交替排列；extract_moments按同顺序选择W的两个1024通道块。W形状(8,2048)重排到(pol,class,channel)，einsum沿channel生成类别空间响应，再重排回(reader,sample,pol,class)。没有把类别轴当样本轴或混用VV/VH。
2. **池化与原特征一致。** 复用同一224原尺度补零、C4图像和footprint旋转、layer3与area footprint权重；每个旋转分别算矩后平均。weighted_moments先减加权空间均值，再减类别公共项并消除数值空间均值，sigma为加权总体标准差，kappa为加权r|r|除sigma。intercept只在均值分数重放时加回，不混入局部矩。并未恢复旧160输入。
3. **OOF配套正确。** load_fold按reader_names明确取outer/inner，使用各模型保存的held全局索引同时装配mean与moments；下游只取train行。实际检查六折held单次覆盖、inner_bin一致、fit/held均属于outer train，以及outer/inner MMSI和pass隔离，全部通过。逐held用同W和features重建的分数与d['oof']一致，未把outer监督矩作为源训练矩。
4. **scaler仅拟合源训练输入。** 校准T用A OOF训练分数；增量用T(A OOF)及同一inner的均值/矩。descriptor首次仅接收train行，验证和predict显式传入保存的mu/scale。score-only按类别坐标分别估计尺度；共享均值/局部头沿样本和类别轴估计各descriptor槽尺度；additive沿用联合15槽缩放，随后分成mean7/local11。均符合被冻结的方法。实际六折接口检查中，验证调用未重新估计或改变mu/scale，输出有限。
5. **valid符合当前数据。** 缓存valid与features完全一致，2746条VV/VH均true；两路实际PNG均读取，footprint按实际大小加权。全true flags经源缩放变为常量0，所有臂一致。本次不把它外推为缺极化数据流程验证，也不把1x1发布chip的valid=true解释为具有充分空间信息。

## 训练与选择

复用的joint_loss确实是普通CE与八类平均CE各半；当前全部outer train和validation八类齐全，不会触发缺类均值或exact_j断言问题。每头200步、AdamW .003/.01，每5步源validation精确分数J选点，含epoch0且仅严格大于才更新，故平局保留最早，没有NLL暗中打破平局。模型最后层为零初始化；实际六折、全部五种调用类型（T与四增量）零输出检查通过。

模型数量/容量对应18个280参数T，以及mean-only209、local209、additive266、extra_score280各18头，共90轨迹。mean-only重复均值槽，不会意外继续使用sigma/kappa；free共享头忽略carrier，因此重复槽中的均值也没有额外乘法含义。additive是两支分数相加并去类公共均值，非新联合交互结构。

全部T选择完成后才训练增量，全部90选点完成后才进入validation/test结果评价。代码提前读取全cohort标签数组，并提前用已冻结T推理全cohort分数，但训练、scaler和选点入口仅传train/validation切片；未发现测试标签进入任何选择路径。预计算固定测试分数不是测试性能反馈，也未形成跨样本归一化。校准模型仅含Linear/GELU，无依赖预测批次统计的层。

必须保留协议边界：T本身在这些源标签上训练，T(A OOF)不是第二层交叉拟合；这是明示设计而非实现遗漏。alpha也由outer源validation统一选择，不是每个inner另行选择。不能将本实现称为整流水线OOF。

## 数值与阅读边界

- 已完成moments_summary.json记录30个reader、64.29秒，均值合成分数对缓存GAP线性分数最大误差2.5608885945160864e-7，top1变化0。
- 本次额外对实际held装配核对，OOF mean两极化求和加各自intercept，对d['oof']最大误差2.5477956788755307e-7；对齐、隔离、源scaler复用及零头检查通过。
- train_heads选点与磁盘head重放均采用float64基分数加float32模型增量转float64，没有旧路径中把验证基分数先截为float32、最终再换回的错配。保存checkpoint后存在源validation J重放断言；最终predict也使用该checkpoint保存的descriptor尺度。
- 阅读范围包括两份主脚本、extract_features及实际被调用的weighted_moments、run_joint、strong/run_integration、standard_controls、run_polarization、run_training相关函数；确认运行导入解析到这些指定模块。没有复审其他历史实验或再做全量XML/原图检查。
- 本审阅未执行训练或重新运行90个选点，不将源码中的重放断言冒充全部训练已经通过的运行回执。现有矩提取数值结果已实际读取；训练是否全部完成由主流程回执记录。

结论：可以按冻结协议继续，无需因本次限定审阅增加臂、修改超参数或重跑reader。后续性能和范围判断交由本轮既定配对统计与原图分析，不从实现通过推导方法必然有效。
