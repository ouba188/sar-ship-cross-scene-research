# 完整源端策略预测：上游交付

日期：2026-10-23。状态：完成。固定seed20261023的一次外部角色划分，训练seed17/29/43；三份策略NPZ已交付，主流程已接入协调器。不再生成额外预测，不训练骨干，不修改历史模型，不Git。

## 交付与接口

- `table.csv`：3387行，含原cohort的row_index、instance_id、obs_id、mmsi、pass_group、port、class_id、role及内部upstream_role。
- `strategies_seed17.npz`、`strategies_seed29.npz`、`strategies_seed43.npz`：统一原cohort行序，`indices[3387]`、`role[3387]`、`logits[3387,3,8]`、`regional_raw[3387,8,69]`。
- 策略轴严格为 `[s0,body,full]`。s0为本轮重拟合的完整A/D/T加two-sided attention，不是layer3 GAP基底。body/full各自以同一个s0加区域残差。
- regional_raw是dynamic的未标准化69维输入；包含s0类别中心化分数、两极化三域mu/h/b及类别中心值、共同32维质量/几何。不得误认成已经按上游scaler标准化的输入。
- `indices.npz`的角色索引为本地table位置；策略NPZ的indices为原cohort位置，二者语义不同，均显式保存。
- 完成及检查记录：`strategies_complete.json`、`IMPLEMENTATION_RECEIPT.json`、`contract_replay_checks.csv`。模型、摘要和原始描述缓存留在本机，不是Git交付物。

## 数据与监督边界

起点是旧fold0的4392源训练+310源验证，共4702条；与三个历史test索引的并集无交集。按pass一次分配55/20/12.5/12.5%，随后按evaluation > selection > utility_train > strategy_fit优先排除跨角色重复MMSI。没有重抽seed，没有按模型结果调整分配，也未触发类别覆盖修复。

|角色|样本|MMSI|pass|类别覆盖|
|---|---:|---:|---:|---:|
|strategy_fit|1646|1290|112|8|
|utility_train|644|587|45|8|
|selection|612|562|28|8|
|evaluation|485|445|29|8|

外部排除1315条，见role_exclusions.csv。上述比例是原pass分配比例，不是排除后的样本比例。四角色pass/MMSI均互斥。evaluation的class7仅3条，selection的class1为24条，不能把3seed当成三份独立评价数据，类均衡指标的样本敏感性需保留。

strategy_fit再用固定规则划出1205 fit_train、296 fit_validation，另145条因内部MMSI冲突不进入拟合或选点。两内部集合pass/MMSI互斥，实际均有8类；fit_validation的class1仅4条。评估函数按实际出现类别计算BA，并不在数值函数中断言8类。

**全部上游监督拟合只用1205条；全部上游选参/选步只用296条。** utility/selection/evaluation的标签只用于划分覆盖记录，不进入任何上游训练、归一化或选点。没有计算三外部角色的性能来选择上游。主流程router另在utility_train拟合、selection选点、evaluation固定评价。

fit内部没有重新搭建每阶段完整OOF：T、attention、body/full训练时使用前级在fit_train上的训练内预测，tau也来自新reader在fit_train上的响应。不能称整个源链OOF。合法性来自效用三个外部角色对本轮所有上游监督拟合/选点整体留出，而不是宣称内部每条预测都OOF。strategy_fit自身预测只供诊断，不得混入协调器效用学习。

## 实际重建内容

保留真实历史结构，重新拟合而不读取历史已训练头充当合法预测：

1. A：缓存R50 stem轴向796维描述经源标准化、RidgeClassifier；alpha在.1/1/10中用内部验证BA/Brier选择。叠加keep_refs五点残差，64→8共享变换后60槽展开→8。
2. D：缓存separated R区域256维浅层描述、2显示均值、2valid标记，260→32→8。五点/D均保留加权MSE、AdamW .001/.001、150步、0/50/100/150选点；seed43的D选择0步，原样保留。
3. T：对U=A+D的8维类别中心化、源标准化分数，8→16 GELU→8残差。不是温度，不是另换layer3分类器。
4. 辅助reader：原layer3双极化2048维GAP及layer1双极化512维GAP各自重新fit StandardScaler/PCA128(randomized17,iter4)/postscale/Ridge，固定logit scale3。alpha由内部验证balancedNLL在1/10/100中选，分别为layer3=1、layer1=100。三训练seed共享这两套确定性reader。
5. attention：原7→12→1均值支路与11→12→1 h/b支路，266参数；保留双极化/C4/全局有效footprint，非moments和attention堆叠。
6. body/full：69→12→1，853参数、零末层、类别中心化输出；body只在新增摘要中清零end/side的mu/h/b，两臂仍共享真实全三区域质量与几何，不是全系统纯body。初始化与共同质量scaler一致。

T/attention/body/full均200步、jointCE=.5普通CE+.5类均衡CE、AdamW .003/.01、每5步内部验证J选点，含0且平局保留最早。合计1个轴向Ridge、2个aux reader及18个神经头，没有嵌套扩大成多套完整上游。

|seed|五点|D|T|attention|body|full|
|---|---:|---:|---:|---:|---:|---:|
|17|150|50|15|0|60|15|
|29|100|50|55|70|30|0|
|43|100|0|105|10|15|0|

**seed29/43的full与s0相同，seed17的attention与B相同。** 这些是真实源选点结果，不能将三策略都描述为活跃且不同，也不能为了给router更多差异改选步。该退化接口及其对效用目标的影响留给主流程分析。

## 冻结前向与mask兼容修复

没有保存大F，也没有重新训练R50。首次fit_train前向同时计算layer1/layer3整图sigma以固定tau；随后对3387条完成摘要前向。每批最多8个对象、至多16幅有效极化图，同一前向同时读取layer1和layer3，模型eval/fp32，C4整90度旋转，原像素160 crop/pad，不resize。layer1沿用stride4/origin0的中心Voronoi面积积分；layer3保留原160→10 area均值语义。

初版核对发现layer1 GAP正常而layer3最大差异约1.14，未放宽容差或带错训练。8个源对象定位显示：新实现旋转后的单通道CUDA mask路径与原三通道mask路径，在180度方向产生最大0.25的area权重差异；stem前后激活无修改，分离的直接前向与原encode也能复现该差异。这里记录实际张量/池化路径现象，不据此推断所有历史CUDA提取均错误。

修复是由原生footprint在CPU上按每个C4方向作精确16×16面积均值，产生10×10权重；这与既定area池化的数学定义相同，没有改变图像、窗口、填充值、有效像素范围或模型。修复后才冻结tau与训练后级。

|已完成核对|最大绝对误差|
|---|---:|
|1205 fit_train layer1 GAP对原缓存|2.38e-7|
|1205 fit_train layer3 GAP对原缓存|5.12e-6|
|3387全保留源 layer1 GAP对原缓存|8.63e-7|
|3387全保留源 layer3 GAP对原缓存|4.58e-6|
|全源layer1 mu之和对W-GAP+intercept|2.49e-14|
|全源layer3 mu之和对W-GAP+intercept|8.88e-15|
|264条重叠样本的原geometry/quality|0|
|real raw69对旧dynamic接口|0|
|保存body/full全量模型重放|0|
|少量不同批量重放|最多2.38e-7，top1一致|

另外两处实施修正：可信本地feature缓存的instance_ids为object数组，读取时允许其既有格式；五点头的连续风险比较因分批数值差异失败，改为同批次严格重放源选点，另验分批分数误差及top1一致，未改变训练/选点标准。初始失败的五点固定轨迹重跑一次，不是追加候选；reader读取修正之后复用已生成A/D/T，没有重训它们。

## 成本与完成范围

- 成功18头的训练、内部选择及重放函数累计13.83秒；两aux reader拟合累计0.63秒。
- 固定温度前向阶段19.53秒，正式全源摘要阶段82.58秒；两阶段中同步计时的骨干GPU前向累计29.28秒，其余包括原图IO、mask、类别投影、传输及保存。
- GPU峰值约258MB，单GPU进程，CPU threads=2。没有大空间F缓存。
- 这些是成功阶段/函数计时，不是用户等待的完整端到端wall time；不包括导入、划分准备、代码调试、失败核对及原历史模型成本。不能写成“完整系统13.83秒”。

本分支完成合法策略数据准备与接口验证，没有评估历史4744目标，也未据evaluation表现改动任何上游。仍是历史开发队列中的源端机制pilot，不是全新独立数据确认。代表原窗结局与router效用分析由主流程完成，本文件不重复制造性能/机制结论。所有本分支exec训练/提取进程已结束。
