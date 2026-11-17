# 后半段联合微调：缓存与梯度路径审阅

2026-11-17。只读审阅新wrapper及实际继承路径，未import模型、未执行GPU/前向/训练、未操作运行中的fits；仅新增本文件，不提交Git。已沿用适用AGENTS与karpathy-guidelines。现有smoke结果作为已落盘证据读取，不声称本代理重新执行过测试。

## 结论

**在实际`layer3 + layer4 + head`范围内，未发现需要停止当前6fits的实质性缓存失效或梯度切断缺陷；没有当前必须补的模型修法。** 这比最初讨论的仅`layer3[5]`更宽，但仍不涉及stem至layer2或原M1/M2/M3参数。32条reference描述子与CCREL source_raw都位于此次可训练后半段之前，原缓存依赖保持固定；CCREL可训练投影的源归一化矩仍按当前权重计算。

这不是全骨干解冻，也不是原M1/M2/M3联合训练。缓存有效、梯度存在只说明实验接入成立，不证明PCEA增益、冻结为原失败根因或训练充分。

## 实际范围与优化器

- [新Model.__init__，第17行](E:/临时会话/paper_draft_20261108/pcea_unfrozen_20261117/model.py:17)：先全冻结，再于23、24、25行打开整个layer3、整个layer4和`net.guide.head`；PCEA组另打开`ccrel`及`exchange`。不存在祖先冻结覆盖新解冻的调用顺序问题。
- [新Model._exchange，第32行](E:/临时会话/paper_draft_20261108/pcea_unfrozen_20261117/model.py:32)：Full臂直接返回conv2原输入args，确实旁路新增PCEA；不是仅冻结PCEA参数但仍使用其更新。
- [新parameter_groups，第35行](E:/临时会话/paper_draft_20261108/pcea_unfrozen_20261117/run.py:35)：明确接收layer3/4、head、reader、exchange，Full仅前两组；67行替换祖先runner的分组函数。祖先原分组仅接收reader/exchange/head，会拒绝新stage参数，但该限制已由wrapper正确处理，不是当前缺陷。
- 读取三份`gradients_*_s17_base.json`，均`passed=true`。各组共有backbone 22,063,104、head 28,684个可训练参数；两PCEA组各另有reader 32,768、exchange 84,163个。Full与PCEA并非总参数量相等，而是共享后半段范围/预算相同，新增PCEA参数明确列出。
- [LiveFull.train，第144行](E:/临时会话/paper_draft_20261108/full_finetune_20261111/train.py:144)在`super().train()`后将所有BatchNorm2d设为eval。layer3/4的BN仿射参数仍可训练，running统计不更新；不能把eval等同于不求参数梯度。

## 哪些缓存需要刷新

| 对象 | 实际依赖与来源 | 本轮结论 |
| --- | --- | --- |
| 原PNG/native像素、footprint、slots、local_slots、区域几何 | 图像、OBB、增强及确定性网格积分 | 不因layer3/4/head训练过期，无需重建。 |
| 32参考的desc/mass/available | stem、layer1、layer2前段及固定anchor projection、几何 | 不受新后半段权重影响。现有训练每microbatch重新提取，已足够；无需增加layer3/4参考特征缓存。 |
| CCREL `source_raw_s17.pt` / 模型source_raw | 固定M1类响应、layer2浅层特征、几何；512维，尚未经过可训练CCREL value投影 | 当前范围可以复用；不因head或layer3/4更新而重建719源bank。 |
| CCREL投影后源token及归一化mean/scale | 当前`ccrel.value`权重 | 不能当永久固定缓存；现有`standardized`已每次重投影并重算矩。 |
| query的layer3/4输出、最终features/logits及历史readout_batch | 可训练后半段和head | 必须live重算；当前训练使用live路径，没有复用旧feature/readout缓存。 |

### 32参考确实止于layer2

[ChipPath.extract，第76行](E:/临时会话/paper_draft_20261108/computational_cost_20261110/benchmark.py:76)计算layer1及layer2前段；第89行用固定`anchor_conv/anchor_bn`形成f2，第90行`pool_multiscale`产生描述子。[reference，第96行](E:/临时会话/paper_draft_20261108/computational_cost_20261110/benchmark.py:96)只返回`desc/mass/available`，没有layer3、layer4或分类head依赖。

[full.loss_for，第172行](E:/临时会话/paper_draft_20261108/full_finetune_20261111/train.py:172)每个训练microbatch调用`net.reference`，再对query执行live forward；并未使用旧静态最终特征。Full与两PCEA组共享这条流程。[LiveFull.finish，第150行](E:/临时会话/paper_draft_20261108/full_finetune_20261111/train.py:150)拼接query/reference描述子后调用网络，并清空临时`readout_batch`。不需要把reference送到新的layer4或重新生成另一个深层参考bank。

### CCREL source_raw仍在固定坐标系

[CCREL.forward，第24行](E:/临时会话/paper_draft_20261108/pcea_coordinate_repair_20261117/model.py:24)先对layer2浅层512维特征做LayerNorm，再依据固定类响应/区域权重池化，44行保存`self.raw`，46行才调用[project_raw，第20行](E:/临时会话/paper_draft_20261108/pcea_coordinate_repair_20261117/model.py:20)投影到32维。它不是layer3/4输出，也不取最终分类head的logits作为类响应。

类响应来自[staged.Network.forward，第43行](E:/临时会话/paper_draft_20261108/staged_ablation_20261109/staged.py:43)的原M1；layer2特征在52行形成，57行进入浅层m2，随后60行才开始layer3。因此本轮更改整个layer3仍不会回改source_raw的输入值。

[coordinate calibrate，第140行](E:/临时会话/paper_draft_20261108/pcea_coordinate_repair_20261117/model.py:140)在no_grad下读取或生成719源预投影bank，并检查源IDs；[PCEA.standardized，第59行](E:/临时会话/paper_draft_20261108/pcea_coordinate_repair_20261117/model.py:59)每次调用当前reader.project_raw，计算当前mean/scale。68行no_grad仅用于复制mean/scale到buffer，71行返回表达式仍使用当次有计算图的mean/scale，不是把整个reader标准化过程detach。

`LocalExchange`局部detail使用复制后的`self.scale`作尺度，因此不经这条buffer路径对尺度求导；这属于既有归一化设计，不会切断CCREL projection、区域摘要或下游stage的全部梯度。不能声称所有统计量路径都参与反传，也不需要为了本轮解冻删除该no_grad。

## no_grad与detach核查

**实际query后段没有整体no_grad包裹。** [staged.Network.forward，第60行](E:/临时会话/paper_draft_20261108/staged_ablation_20261109/staged.py:60)在线执行layer3前五块；62行进入末块conv1，68行调用conv2并触发PCEA pre-hook，69行conv3残差，80行在线执行layer4，93行进入guide/head。前缀参数固定只让前缀输出不需要梯度，不妨碍后续有requires_grad参数的卷积重新建立计算图。

容易误判的三处：

1. [BottleneckEvidenceResNet50.stage3_paths，第105行](E:/临时会话/sar_m123_standalone/sar_m123/bottleneck_evidence.py:105)及其他祖先有受保护比较路径的no_grad，但当前staged.Network手写上述后段，不调用这条stage3_paths；不能凭搜索到no_grad就判定新stage不训练。
2. [LocalExchange.forward，第115行](E:/临时会话/paper_draft_20261108/pcea_local_cross_20261117/model.py:115)的区域RMS `detach`只冻结幅值归一化因子的梯度；`x+delta`、`local_x`、a/b和下游卷积仍可求导。trace中的detach只供诊断，不替代真实返回值。
3. [full.predict，第160行](E:/临时会话/paper_draft_20261108/full_finetune_20261111/train.py:160)与source calibrate使用no_grad是预测/固定源统计路径；训练[loss_for，第172行](E:/临时会话/paper_draft_20261108/full_finetune_20261111/train.py:172)无该装饰器。冻结原M2参数也不等于冻结它对输入的导数；[RegionPoolWriteback.forward，第28行](E:/临时会话/sar_m123_standalone/sar_m123/bottleneck_evidence.py:28)仍计算`x+update`，无输入detach。

## 已有证据与边界

读取[scope_check.json](E:/临时会话/paper_draft_20261108/pcea_unfrozen_20261117/scope_check.json)：receiver最大变化0.0000100433826；reference、source_raw、prefix及BN running变化均为0。[smoke，第71行](E:/临时会话/paper_draft_20261108/pcea_unfrozen_20261117/run.py:71)在94/95行分别断言layer3[5].conv2和layer4[2].conv3梯度非零。与静态依赖分析一致，未见“名义解冻但没有接收梯度”的证据。

该scope smoke的`source_raw_change`是同一12条源输入重新算出的预投影raw差异，不是重新扫描全部719条bank；`prefix_change`取layer2[3].conv3代表权重，其他范围由requires_grad清单和代码依赖支持。它不能证明全训练历程、未来新增模块或所有增强都已被测试，但当前没有新失败证据要求重复启动GPU检查。

范围扩大后的最小处理，**不是要求当前补改**：

- 若再解冻stem/layer1/layer2前段，reference描述子及CCREL raw会随权重变；保留当前live reference提取，并改为按当前前缀重算源raw，不能继续盲用只按seed命名的旧bank。
- 若只解冻M1或layer2末段的live conv3，至少重审/刷新CCREL source_raw；reference使用独立anchor conv3，其是否变化应按具体上游参数判断，不能一概宣称所有缓存都失效。
- 当前scope无需删除calibrate的no_grad、无需每步重算719原始源特征；最小修法就是维持新wrapper已经实现的解冻和分组，不额外改模型。

后续比较必须使用本轮同预算、同解冻范围的新Full。PCEA组训练后的disabled仅移除该组新增分支，保留该组已协调更新的layer3/4/head；它既不等于独立训练的本轮Full，也不等于原始冻结Full。三组共同的微调收益不能归功于PCEA；本次审阅尚未读取训练收益或原图新结果。
