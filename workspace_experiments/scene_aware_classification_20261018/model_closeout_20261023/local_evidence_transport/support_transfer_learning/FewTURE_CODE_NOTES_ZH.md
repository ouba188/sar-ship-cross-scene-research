# FewTURE 官方 token reweighting 代码依据与本轮适配边界

日期：2026-10-24。范围：本轮 `support_transfer_learning` 的已知方法基线依据与限定实现核对，不设计新模块、不重新训练、不操作 Git。

## 结论

官方实现可以支持“用支持图像之间的预测监督优化支持 token 重要性”这个基线，但本轮只能称为 **FewTURE-inspired normalized support weighting**。冻结 R50/PCA32、VV/VH、20 个几何槽位、区域内匹配、归一化质量、跨过境屏蔽，以及 Adam/200 步均为本地适配；不是 FewTURE 官方复现，也不能据此评价原论文完整方法的优劣。

尤其需要区分两个实现细节：官方权重加在温度缩放之前；官方将一类的全部支持 token 与一张查询图的全部 token 对一次性 LogSumExp，而不是逐查询 token 算分类 CE 再平均。

## 1. 实际读取的来源

已先读取上一级 `competitive_matching/NEXT_ALGORITHM_ZH.md`，限定本轮为来源明确的对照，不将普通支持重加权命名为第三强创新。

作者官方仓库：`https://github.com/mrkshllr/FewTURE`。

读取版本：`c67d4c1fe01d683abcc46eb409ce8eed6fcf1b76`。GitHub API 一度限流，后通过 GitHub 页面内嵌的 `refInfo.currentOid` 取得提交，再以该提交的 raw 文件逐文本核对；下列三个文件与本次读取的 main 相同。未下载模型，未安装环境。

| 依据 | 核对位置 |
| --- | --- |
| [eval_FewTURE.py](https://github.com/mrkshllr/FewTURE/blob/c67d4c1fe01d683abcc46eb409ce8eed6fcf1b76/eval_FewTURE.py) | 67–85 默认值；143–179 token/余弦；183–265 PatchFSL；319–350 checkpoint 温度；384–410 冻结与在线适配 |
| [train_metatrain_FewTURE.py](https://github.com/mrkshllr/FewTURE/blob/c67d4c1fe01d683abcc46eb409ce8eed6fcf1b76/train_metatrain_FewTURE.py) | 57–105 参数；261–345 内环；450–478 外环优化器；501–512 外环梯度；547–551 保存温度 |
| [README.md](https://github.com/mrkshllr/FewTURE/blob/c67d4c1fe01d683abcc46eb409ce8eed6fcf1b76/README.md) | 61–82 自监督预训练；156–181 meta fine-tuning/评估命令；193–194 许可 |

还读取 `models/head.py`，确认其主要是 DINO/iBOT 预训练 head，不是 token reweighting 所在文件。官方 token 重加权位于上述两个主脚本各自的 `PatchFSL`。仓库许可证为 Apache 2.0；若实际复制代码需保留适用版权与许可说明，本笔记仅解释公式与行为。

## 2. 官方推理算子究竟是什么

设一轮有 C 类，每类 K 张支持图，每图 P 个 token；查询图 q 有 Q 个 token。代码将支持排列为 class-major，再展平 image/token。

支持 token s 与查询 token t 的相似度为各自 L2 归一化后的余弦 `C_st`，归一化分母下限 `1e-8`。ViT 路径去除第 0 个 CLS token，读取其余 patch token。没有在这里额外训练一个局部投影头。

每个支持 token 有一个标量 `v_s`。第 c 类对查询图 q 的最终 logit 是：

```math
L_c(q)=\log\sum_{s:y_s=c}\sum_{t\in q}
\exp\left((C_{st}+v_s+M_{st})/\tau\right).
```

训练支持权重时使用遮挡项 M；真正查询推理时 M=0。对这一整张图的 C 维 logits 做 CE/argmax。

具体含义：

- `v` 的长度是 C×K×P，零初始化、无界可正可负，是 leaf tensor，显式加入优化器；不是一个由 query 网络生成的注意力，也不是查询 token 权重。
- 同一个支持 token 权重广播给所有查询 token 与所有查询图。它没有“针对每个候选类别重新挑一套 query 权重”的过程。
- 权重没有先做 softmax，没有逐类重新归一，也没有 sigmoid/cap/显式质量守恒。其指数作用相当于乘上 `exp(v_s/tau)`。
- 官方依次 reshape/transpose 后，对 `K×P×Q` 维做一次 LogSumExp。因此不是 `mean_t logsumexp_s(...)`，更不是 `mean_t CE(...)`。
- 将一类所有支持权重同时增加常数 a，会把该类 logit 增加 `a/tau`。所以官方自由权重也能表达类别分数偏置。本地类内质量归一化主动去掉了这项自由度。
- 默认每类相同 K、每图相同 P/Q，原始求和计数较一致；代码没有为可变槽位/缺失极化做数量修正。不能把零向量占位当作本轮的合法缺失处理。

## 3. 温度、优化器、步数和 reduction

| 项目 | 官方实际代码 | 本轮适配 |
| --- | --- | --- |
| 温度 | 默认 `0.051031036307982884`，数值等于 `1/sqrt(384)`；它是固定默认数，不根据 embedding 维数自动更新 | `1/sqrt(32)`，约 0.176777，是本地选定值 |
| 温度作用位置 | `(cosine + v + mask) / tau` | `cosine/tau`，另加归一化后的 `log mass`；v 是 log mass 单位 |
| 评估温度 | `log_tau_c.requires_grad=False`；metaft checkpoint 的 `temp_sim` 若存在，会覆盖命令行温度 | 没有使用官方 checkpoint 的温度 |
| 权重优化器 | SGD，只传 v 与 lr，默认无 momentum、无 weight decay | Adam，lr=0.03；额外 `0.01*mean(v^2)` |
| 在线学习率 | 0.1 | 0.03，不是沿用官方优化尺度 |
| 在线步数 | 15；每步全部支持图互相分类 | 200，保存 0/15/50/100/200 |
| 内环 loss | `CrossEntropyLoss()`，默认 mean；每张支持图一个 C 维预测，平均 C×K 张图 | 同样是图像级 CE mean；64 张源参考作为 pseudo-query |
| token reduction | logits 内部对全部 token 对 LSE，loss 不再除 P/Q | 区域内 query 面积归一、support 质量归一；六组分数平均 |
| 权重重置 | 每个新任务从零开始；官方 eval 每个 episode 创建 PatchFSL | 当前固定源64参考库训练一次，向不同源角色的查询迁移 |
| 选步 | 官方 eval 按固定15步，不读测试 query 标签选步 | utility644 用局部 J 选 checkpoint，属于额外的源验证监督 |

优化尺度不能只按数值学习率比较。若把官方写成 `cosine/tau + alpha`，有 `alpha=v/tau`；在无其他改动的 SGD 下，等价 alpha 学习率是 `lr_v/tau^2`。官方默认 tau 对应约 38.4，而不是 0.1。**本轮还有类内归一化和 Adam，不应使用这个关系宣称两者优化等价**；这个推导只是防止错误移植权重位置。

支持数、有效 token 数、归一化和图像级 mean 都影响每参数获得的梯度。使用 64×2×20 个本地参数时，没有依据要求官方15步必然足够，也没有依据将本地200步称为官方配置。

## 4. 官方 mask 的准确行为

### 多 shot

K>1 时，`block_diag` 为每张支持图生成 P×P 的 `-100` 方块。该图作为 pseudo-query 时，**本图所有支持 token 全部被屏蔽**，而非只去掉相同 token。其他同类图保留；不同类的其他图也保留。

mask 加在温度之前，所以默认相当于 logits 加约 -1959.6，而非 -100。官方是有限负数，非严格 `-inf`。

### 单 shot

K=1 时不能屏蔽本类唯一整张图，因此使用展平 token 索引的对角带。默认 `block_mask_1shot=5` 的代码屏蔽 `|row-col|<5`，即内部通常9个对角位置，不是5个像素宽，也不是图像上的5×5空间邻域。它在整个展平矩阵上作带状遮挡，图像边界附近也受索引关系影响。

**本轮是8参考/类，不应搬这个单 shot 遮挡，更不能把有语义的20个 OBB槽位当作规则栅格，用索引相邻等价空间相邻。**

### 本轮区别

本轮 image 臂实际按 MMSI 不同构造 mask。由于参考库当前每图对应独立 MMSI，实测等价于屏蔽同图；未来若同船多图进入库，则会是更严格的同船屏蔽，应重新说明名称。pass 臂按完整 `pass_group` 屏蔽同过境，不读取 pseudo-query 类别来决定哪些链接允许。

官方没有 pass、地理港口、VV/VH 或 OBB区域语义。不能把跨过境屏蔽归为作者已有代码行为，也不能只靠这个任务 mask 的改动宣称强算法创新。

## 5. 与骨干 meta-training 的关系

官方链路不是仅在任意冻结特征上拟合15步 v：README 提供自监督 masked-image 预训练、episodic meta fine-tuning 和最终评估三个阶段，也允许评估 pretrained-only checkpoint。

官方 meta fine-tuning：

1. 骨干对 episode 的支持/查询图提取 token。
2. 内环先将支持 key/query embeddings `detach`，只用支持标签优化 v。v 的 SGD 不构成可微展开的二阶内环；不沿 v 的优化历史把梯度传回骨干。
3. 内环结束后用原本未 detach 的支持与查询 embedding 预测 query，query CE 外环更新骨干。外环优化器不含 v。
4. 默认外环 SGD lr=0.0002、momentum=0.9、Nesterov=True、weight_decay=0.0005。可选 `meta_learn_similarity_temp` 默认为 False；True 时把 log tau 加入外环参数，lr=25×meta_lr。内环优化器仍只有 v。
5. 最终 eval 冻结骨干，支持数据启用梯度仅用于优化 v，测试 query 标签只用于准确率，不参与 v 学习。

故“冻结 R50/PCA32 的支持重加权没有提高完整系统”不能推出 FewTURE 的 Transformer/meta-trained 表征无效；反过来，即使当前基线提高局部 OA，也只支持本地输入与协议下的支持监督有用。

## 6. 当前归一化算子的实际含义

`run.py` 将20槽分成 body8/side8/end4，并对 VV/VH 分开形成6组。组 g 的查询面积质量 a 与每条参考内部的支持面积质量 b 先分别归一。

```math
K_{qgcrj}=\log\sum_i a_{qgi}\exp(C_{qgicrj}/\tau),
```

```math
\rho_{qgcrj}=
\frac{I_{qr}b_{gcrj}\exp(v_{gcrj})}
{\sum_{r',j'}I_{qr'}b_{gcr'j'}\exp(v_{gcr'j'})},
\qquad
L_{qgc}=\log\sum_{rj}\rho_{qgcrj}\exp(K_{qgcrj}).
```

其中 I 是同图/过境允许关系，推理时在合法且独立的源参考库上全部允许。reference 臂的 v 只依赖 c、r，在极化/区域/slot 间共享；token 臂逐参考、极化、slot 设 v。最后按有效组数平均，再对类中心化。

两次 LSE 在代数上仍是归一化 token 对的加权指数和；不是两次相互独立的分类，也不是先获得每个 query token 类概率再平均。

直接可复用的 FewTURE 思想与操作：局部余弦、支持 token 的全任务共享可学习权重、支持互相预测、屏蔽自引用、图像级 CE、v=0 的无学习对照。

必须单独声明的适配：源PCA32和几何槽位、同区域而非全图 token 任意匹配、两个极化及有效性、面积/类内质量归一、跨过境 mask、reference 自由度对照、Adam/正则/200步、utility选步、最终残差融合。尤其类内归一化改变了算法假设，不是纯数值稳定化。

推荐报告名称：**FewTURE 启发的源支持预测重加权基线（normalized, frozen-feature adaptation）**。本轮64图是源参考库，不是64个目标标注支持案例。

## 7. 本轮代码关键风险限定核对

仅核对 `run.py`、`PROTOCOL_ZH.md` 及已生成缓存，不重跑训练/分析，不扩到其他模块；不替主代理重做逐类和原图分析。以下检查用于发现会改变当前结论的索引/遮挡/泄漏/归一化实现错误。

### 已确认

- `load()` 的 instance_id 顺序与 token cache 一致；bank 为8×8、64条不重复，bank 每行正确对应0–7类。`kernels` 的 `bckij` 和 Readout 的 class/reference/slot 排列相符。
- bank 实测64图、64船、35过境，全部 `strategy_fit`。与 utility/selection/evaluation 的 MMSI、pass_group 集合交集均为0。bank 的 MMSI/pass 无缺值。当前推理 `allowed=ones` 不会读到同船/同过境参考。
- 权重梯度的标签来自 `labels[bank.ravel()]`；utility644 标签只在 checkpoint 的局部J选择中使用；selection612 只用于融合 gain；evaluation 标签只用于最终指标/配对记录。utility不是无标签数据，不可写成“总共只用了64个标签完成全部模型选择”。
- 固定 kernel 可同时缓存所有源角色，因为 kernel 不用查询标签且本地骨干/PCA冻结。这里未重新审计上游 PCA 拟合来源，只沿用已冻结源PCA的实验前提。
- query a 的面积归一沿 query-slot 维；支持 b 先在每参考/区域内归一，随后 v 重加权后在同类的参考×slot 上归一。因此零权重时是有效参考等质量；不能把它误说成所有 slot 天然等权。
- 参考权重在六组共享，token 权重按极化及区域切片，end4填充至8的位置原质量为0。有效组由 query非空且该类有可用支持共同决定；无效组不进入 `Readout.__call__` 平均分母，不是以零分参与六组固定平均。
- 所有臂的融合尺度共用 uniform 在 utility644 的类中心化RMS；没有用evaluation自适应RMS。公共倍率对照使用uniform在selection选出的倍率，符合协议。
- image/pass屏蔽都作用于所有候选类别，不是用真类只屏蔽正样本。有限负数 `-1e4` 用于当前缺失mask；应以当前实际权重范围确认其不会被抵消，不把抽象的无界权重风险当本轮错误。

只读缓存数值核对结果见下节；若未发现实际问题，不因它不是官方复现而要求重训。本轮 OA/BA 的取舍与最终增益由主代理正在进行的类别和原图分析解释，不在本支线重新选择配置。

## 8. 核对结果补记

只读缓存核对已终态完成，未重跑主训练/主分析，未生成新的分类结果。没有在上述限定范围发现实际实现错误。

| 项目 | 实测 |
| --- | --- |
| image 训练屏蔽 | 64条有向图对，恰为自身；对角全部禁止；每类最少剩7参考 |
| pass 训练屏蔽 | 184条有向图对；对角全部禁止；每类最少剩5参考 |
| 支持训练有效性 | image/pass相同：VV每组64×8=512有效 query-class，VH每组56×8=448 |
| 缺失VH | 64参考中8条；1741查询中203条。缺失query的VH组从平均分母剔除 |
| 质量 | mass全有限且非负；所抽查询的每组可用性在各类一致 |
| 数值重放 | 五臂抽样查询与已保存 local_scores 完全一致 |
| 独立标量归一核对 | 3条查询，含缺失VH；五臂最大绝对误差 `4.7182e-7` |
| 已选权重绝对值上界 | 五臂最大4.4602，远小于mask常数10000；当前不存在有限mask被权重抵消的问题 |

标量核对逐class/reference/slot从原mass、kernel与已保存weights重新计算分母及加权指数和，不调用 `Readout` 的归一化函数来构造“期望值”。这检验所选实际案例的索引与公式一致性，不宣称已穷举所有输入或全面审计上游。

## 9. 追加2×2 mask的标签边界

另只读核对当前 `run.py --factorial` 的新增分支（约139–148行、245–253行）：

- `y` 仍只来自64条bank的已知源类别；`same_class` 比较的是这64条参考的标签，不是1741条utility/selection/evaluation查询标签。
- `token_crosspos` 删除同过境且同类的训练边，另去除自图；总90条，相比image额外26条。
- `token_crossneg` 删除同过境且异类的训练边，另去除自图；总158条，相比image额外94条。
- 原pass臂删除两部分，即64+26+94=184条。统计为有向图对，不能当120个独立观测或独立船对。
- 两臂类别相关mask只传给 `train_reader`。统一推理 `reader` 在循环前创建，使用 `np.ones((len(idx),8,8))`；未复用训练mask，没有以查询真类选择参考。
- 新结果输出在 `mask_factorial`；原始kernel只读复用，ids/bank需一致，原结果不被覆盖。

**判断：该追加是有标签的源训练边分解，不构成查询标签泄漏，也不是部署时要求知道query类别。** 原pass屏蔽本身符合原协议；此2×2实验用于区分正/负训练边删除的效果，不应在尚无证据时称原实现有bug。已完成边界核对，不在本支线重复训练、逐类指标或原chip分析。
