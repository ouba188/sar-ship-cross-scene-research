# CPINet 独立复现：算子、维度与实现边界

日期：2026-11-11。用途：本轮 OSS2 四分类强基线实施与独立复核。本文只记录已经核查的来源和当前代码，不继续扩展检索。

## 1. 当前结论与状态

采用 SAR-DenseNet 型双独立骨干、IMDFF、MO-SE、四路 FBC、共享分类 FC 和 GradNorm 的完整 CPINet 方法路径，在当前 native160、固定划分和极化可用性协议下独立实现。不是共享 R50 版本，也不是作者官方代码或 exact 原实验复现。与 Full 的预训练、骨干容量、全量训练及优化预算不同，应作为系统级对照解释。

已明确纠正的原文偏差是 MO-SE excitation 隐层：原 PDF 第10页图3标注 C/2，先前代码为 C/16。本次落笔前只读确认 `cpinet.py` 的 `reduction=2` 已生效；136通道对应68维隐层。用户报告已经停止 PID4556，并将原运行保留为 `implementation_deviation_c16_s17`，不读取其测试结果。该进程状态和目录处置按用户报告记录，本支线没有操作训练或独立检查进程。

初始通道、TL卷积/压缩、GBP具体组数等未核实项不再阻塞实施。冻结当前合理选择并明确披露即可；不能把“当前实现可运行”写成“作者配置已完全还原”。此前讨论过的初始30通道、末级165通道只是候选假设，并无足够原文证据，本报告不将其作为实施要求。

标记约定：

- **P：原文明确。** CPINet 正文、公式或图中直接给出。
- **D：维度推导。** 从明确结构及本轮输入/代码选择计算得到，不是作者额外报告。
- **R：引用组件来源。** 来自被引方法的官方实现或相关一手材料，不自动证明 CPINet 作者采用了完全相同实现。
- **C：本轮选择。** 原文未明确、未取得必要来源，或为了当前数据协议而作出的实现决定。

## 2. 来源与阅读边界

**[P1] CPINet 官方 HTML：** `https://www.mdpi.com/2072-4292/16/18/3479`，DOI `10.3390/rs16183479`。已核对第3节式(1)-(9)、第4.1/4.2节、参数与消融相关内容。

**[P2] 用户提供的原英文 PDF：** `E:/Docms/论文实验/给老师版本/MSPNet_IEEE_LaTeX_Source_20260626_102230/downloaded_references/PDFs/23_ref23_J_He_2024_CPINet_Towards_a_novel_cross-polarimetric_interaction_network_for_dual-polarized_SAR_ship_c.pdf`。已实际查看第7页图1、第8页图2、第10页图3渲染；阅读第11页 FBC/图4文字及第12-14页优化和实验设置。图1/2没有给出初始通道或TL压缩率。图3给出了 excitation 的 C/2 隐层，但未消除 GBP 输入宽度的所有省略。

**[R26] 参考文献[26]：** He et al., *Group Bilinear CNNs for Dual-Polarized SAR Ship Classification*, IEEE GRSL 19, 4508405, 2022，DOI `10.1109/LGRS.2022.3178080`。IEEE入口及PDF入口未成功取得可读全文；不能声称核对过该文架构表图。

**[R30] 参考文献[30]：** He et al., *Multi-Scale Dense Networks for Ship Classification Using Dual-Polarization SAR Images*, RadarConf23, 2023，DOI `10.1109/RadarConf2351548.2023.10149595`。同样未取得可读全文。MSDFF本轮依照 CPINet 自己重述的式(1)、图2实现，不声称完整核验[30]。

**[R26a] 同作者相关专利：** `https://patents.google.com/patent/CN114973016A/en`，PDF `https://patentimages.storage.googleapis.com/89/b5/f2/ee5a2bc56983ba/CN114973016A.pdf`。已读结构说明并查看附图。其骨干描述包含两层 Conv-BN-ReLU 基础层，以及 BN-激活-卷积-平均池化的TL；GBP描述包含逐元素平方根与L2处理。它是补充的一手技术材料，不能替代[26]原文，更不能据此认定 CPINet 的 stem、TL或GBP归一化确切配置。

**[R33] FBC 原论文及官方代码：** Gao et al., *Revisiting Bilinear Pooling: A Coding Perspective*, AAAI 2020。作者PDF `https://wu-yuwei-bit.github.io/paper/AAAI2020_Gao.pdf`；论文给出的仓库现重定向到 `https://github.com/zhigao2017/FactorizedBilinearCoding`。已核实 `models/FBC.py` 第60-68行：稀疏编码、平均池化、有符号平方根、L2归一化、线性分类；`models/SC.py` 第19行是 beta/2 软阈值。原始文件：

```text
https://raw.githubusercontent.com/zhigao2017/FactorizedBilinearCoding/master/models/FBC.py
https://raw.githubusercontent.com/zhigao2017/FactorizedBilinearCoding/master/models/SC.py
```

**[R35] GradNorm 原论文：** `https://proceedings.mlr.press/v80/chen18a/chen18a.pdf`。已读第3节及Algorithm 1；确认目标梯度停止求导、只对任务权重求GradNorm梯度、权重和归一到任务数，以及原算法按训练step更新。

**[L1] 本地已有记录与代码：** 本目录 `BASELINE_READINESS_ZH.md`、`cpinet.py`、`train.py`、`data.py`、`IMPLEMENTATION_LOG_ZH.md`。已读根目录和论文目录AGENTS，以及 `codex-brain/INDEX.md` 和相关工作流。此次用户的只读审查、仅写指定报告、不提交Git约束优先于通常研究同步流程。

## 3. 骨干：明确部分与冻结选择

| 项目 | 状态 | 本轮规范及依据 |
| --- | --- | --- |
| 双极化骨干 | P/C | P：pseudo-Siamese；C：两个结构相同、参数独立的骨干及IMDFF。明确共享的是后端FC，不把它推广为骨干共享 |
| DB数量/层数 | P | 5个DB，每个3个稠密连接卷积层；增长率依次3、6、9、12、15 |
| 卷积与空间保持 | P/C | P：骨干卷积核3×3；C：stride1、padding1，当前Dense层为BN-ReLU-Conv-Dropout2d(0.2) |
| 多级取点 | P | 前四个DB后的TL输出，以及第五个DB输出；不是全部取池化前 |
| 下采样 | P | 前4个DB后2×2平均池化；本轮stride2 |
| 初始通道/stem | C，待定 | 当前单通道直接进入DB1，Identity stem。未查实作者原版stem和初始通道 |
| TL卷积/压缩 | C，待定 | 当前仅AvgPool2d，无TL卷积、无通道压缩。不能称[26]明确配置；也不能按一般DenseNet习惯擅称1×1、theta=0.5为原版 |
| Dropout种类/位置 | C | 原文给rate0.2，但未充分规定每处位置及channel-wise/element-wise种类；当前Dense层Dropout2d及FC前Dropout须披露 |

当前每个Dense层执行 `x <- concat(x, H(x))`。不压缩时，第i个DB输出通道递推 `C_i=C_(i-1)+3*g_i`，`C_0=1`。因此有下表；空间尺寸依赖本轮native160和保持分辨率的卷积设置。

| 输出 | 通道 | 160输入尺寸 | 原文64输入下同一配置尺寸 |
| --- | ---: | ---: | ---: |
| F1：DB1+TL1 | 10 | 80×80 | 32×32 |
| F2：DB2+TL2 | 28 | 40×40 | 16×16 |
| F3：DB3+TL3 | 55 | 20×20 | 8×8 |
| F4：DB4+TL4 | 91 | 10×10 | 4×4 |
| F5：DB5 | 136 | 10×10 | 4×4 |

**D：DB4与DB5同空间分辨率。** 不能把五级机械写成80/40/20/10/5。当前代码的累积stride `(2,4,8,16,16)` 正确。

## 4. IMDFF 精确计算

对每个极化独立计算，令 `f` 表示 `Conv1x1 -> BN -> ReLU`（P）。第3级融合严格按式(1)：

```text
F3' = F3 + f3(concat(maxpool4(F1), maxpool2(F2), up2(F4), up2(F5)))
```

concat不包含目标F3自身；F3走残差相加。上采样明确为转置卷积，不是默认双线性插值（P）。当前选择 `ConvTranspose2d(C,C,kernel=2,stride=2,padding=0,bias=False)` 保持源通道（C）；转置卷积核、padding及输出通道没有从已读原文中查实。

按相同融合逻辑实现第4/5级（P支持多级融合；下列具体同尺度处理是C）：

```text
F4' = F4 + f4(concat(maxpool8(F1), maxpool4(F2), maxpool2(F3), F5))
F5' = F5 + f5(concat(maxpool8(F1), maxpool4(F2), maxpool2(F3), F4))
MS  = fMS(concat(maxpool2(F3'), F4'))
```

F4/F5同尺度支路当前直接Identity。三个融合目标均使用原始五级输入，不把先算出的F3'递归注入F4'/F5'。本轮维度：

- `f3: 265 -> 55`，输出 `B×55×20×20`。
- `f4: 229 -> 91`，输出 `B×91×10×10`。
- `f5: 184 -> 136`，输出 `B×136×10×10`。
- `fMS: 146 -> 136`，输出 `B×136×10×10`。式(2)没有明确fMS输出宽度，选136是C，便于四个FBC头使用同一输入宽度。

## 5. MO-SE：GBP、隐层和跨极化方向

输入是各极化的 `F5'`，不是原始F5，也不是MS（P）。设 `X` 为 `B×C×H×W`，`N=H*W`，等分G组，每组d=C/G通道。

一阶描述子为 `GAP(X)`，维度 `B×C`（P）。二阶式(5)枚举 **所有 `1<=i<=j<=G` 组对**，包括同组和不同组；每对保留完整 `d×d` 矩阵。不能只计算G个同组Gram矩阵，也不能只取每个同组矩阵的上三角。

当前实现的可执行定义（C：空间取平均且不做额外power/L2）：

```text
Xi: B×d×N
Bij = Xi @ Xj.transpose(-1,-2) / N
GBP = concat(flatten(Bij) for i=1..G for j=i..G)
D_GBP = G*(G+1)/2 * d*d
```

本轮 `C=136,G=8,d=17`，36对，`D_GBP=10404`（D）。G=8不是已核实的原论文设置；GBP空间求和/平均及power/L2也未从CPINet正文查实。相关专利中的归一化不能直接提升为CPINet事实。当前使用平均、不加额外归一化，须明确记录。

excitation按图3与式(3)：`Linear -> BN -> ReLU -> Linear -> BN -> Sigmoid`。图3明确中间为C/2；当前实现选择GAP和GBP路径都输出C维门控，并都采用C/2隐层：

```text
GAP path: 136 -> 68 -> 136
GBP path: 10404 -> 68 -> 136
```

图3把excitation输入示意标为C，但式(5)产生的GBP实际宽度通常不是C。因此“GBP首层直接接10404维，再输出68维”是与公式相容的维度补全（C）；不额外发明未经报告的GBP降维投影。四个excitation网络（两极化×两阶）当前参数独立（C）。

两阶分别经过sigmoid后相加，门控范围为(0,2)，再乘到**另一极化**（P）：

```text
gateVV = exciteGAP(VV) + exciteGBP(VV)
gateVH = exciteGAP(VH) + exciteGBP(VH)
augVV  = F5'_VV * gateVH[...,None,None]
augVH  = F5'_VH * gateVV[...,None,None]
```

不是相加后再sigmoid，也不除2、不加额外 `1+gate`。当前代码方向正确。C/16旧运行属于已知实现偏差，不能混入C/2主结果。

## 6. FBC：式(7)及归一化出处

P：`k=2048,r=1,lambda=0.001`。给定同尺度两输入 `X,Y: B×p×H×W`，两个可学习投影 `U,V: p -> r*k`；P矩阵按连续r个元素求和。r=1时P为单位映射，不必显式构造：

```text
A = Conv1x1_U(X) * Conv1x1_V(Y)      # B×2048×H×W
S = sign(A) * relu(abs(A)-0.0005)
z = max(S, spatial dimensions)       # B×2048
```

阈值为lambda/2，不是lambda。这里是有符号逐元素最大池化，不是max(abs(S))、平均池化或只保留正值。自融合使用同一特征作X/Y，但U/V仍是独立投影；不等同于直接平方。训练直接实现式(7)，不逐批求解外部LASSO，也不额外把式(6)重构损失加入四路CE。

R/C：CPINet式(7)/图4没有写出pooled code的power/L2；[33]官方 `FBC.py` 明确在pool后做signed-sqrt和L2再分类。本轮采用：

```text
q = sign(z) * sqrt(max(abs(z),1e-8))
q = L2_normalize(q, channel_dimension)
```

保留CPINet明确的max，不照搬[33]官方代码的average。1e-8是本轮零附近数值处理；四个FBC的U/V独立、投影无bias属于C。[33]原实现用带bias的Linear，因此不是逐行复制该仓库。这里归一化的是FC输入向量，不是分类器权重；不附加FC权重归一化、cosine分类器或额外温度。

失败及修正记录：主代理记录原始未归一FBC在epoch0/step5出现非有限损失，初始GradNorm权重梯度约111-193；该负结果保留。用户随后报告归一化后四头CE约1.38、权重梯度约0.49，2epochs/46steps无NaN。这些是主代理验证结果，本支线未重复运行，也不能据此证明唯一根因或长期训练稳定。

## 7. 四路头、缺路与集成

| 分支 | FBC输入 | 有效条件 |
| --- | --- | --- |
| DB34 | MS_VV, MS_VH | VV和VH均可用 |
| MO-SE | augVV, augVH | VV和VH均可用 |
| DB5-VV | F5'_VV, F5'_VV | VV可用 |
| DB5-VH | F5'_VH, F5'_VH | VH可用 |

P：四路FBC全局表示、共享FC、四个CE。C：本轮直接使用同一个 `Linear(2048,4)`，FC前Dropout0.2；不存在四个独立分类器。原文没有充分明确额外embedding宽度/输出层设计，因此直接2048到4是本轮选择，不将相关专利的128维FC硬套入CPINet。

每头CE只对有效样本取均值。推理对有效头的softmax概率等权平均；原文只说集成并指向既往文献，等权概率平均是C，不能称已核实的作者集成公式。GradNorm损失权重不自动作为推理权重。

当前缺路实现：缺失图像置零；两个骨干及注意力仍全批前向，然后屏蔽不可用头的CE及集成贡献。**训练态BN仍会受到补零分支影响**，包括有效样本的归一化与running statistics。这是代码中确实存在的统计影响，不是标签泄漏或已证明的性能退化。eval模式改变缺失图像而输出不变的测试，仅证明推理头屏蔽，不能验证训练期隔离。当前可作为C披露；如以后改为按可用样本前向，应单独记录，不能悄悄混入同一训练配置。

## 8. GradNorm可执行规范

P：在共享FC的权重W上计算梯度，不在两个独立骨干上替代计算；alpha=0.12；目标梯度视作常量；四个损失权重和归一到4。

```text
Li       = masked mean CE for head i
ni       = stopgrad(norm(grad_W Li, 2))
Gi       = wi * ni                         # wi保持正值
ratio_i  = stopgrad(Li / L0_i)
ri       = ratio_i / mean(ratio)
target_i = stopgrad(mean(G) * ri**0.12)
Lgrad    = sum(abs(Gi-target_i))
```

对Lgrad只求wi梯度并更新wi；正值裁剪后使其和为4。随后用 `sum(stopgrad(wi)*Li)` 更新模型。将ni detach不会丢失这里需要的权重梯度，因为正wi下范数对wi是线性的；不需要为了更新wi给整个网络建立二阶梯度。

已读取的当前 `train.py`：第一epoch保持wi=1，积累各batch的未加权头损失，epoch结束以batch均值写入L0；第二epoch起逐minibatch更新权重。第一epoch均值取法和预热是C；它尊重CPINet的首epoch基准描述。CPINet叙述按epoch更新，而[35]Algorithm 1按step更新，所以当前minibatch频率须披露，不冒称与CPINet描述逐字一致。有效头样本数不同时，当前“batch均值的均值”不等于整epoch有效样本加权均值，报告采用前者即可。

本轮模型优化器SGD momentum0.9、weight decay5e-4；wi另用SGD momentum0.9、不施weight decay。模型和wi学习率当前相同并一起按epoch衰减。原文没有充分区分两套优化器的细节，wi无weight decay和正值下限1e-3是C。未归一阶段的大权重梯度与学习率会造成大幅更新；修正后不得把旧不稳定指标当成当前配置仍然失败的证据，也不因已稳定就无依据追加更多归一化。

## 9. 本轮数据、预算与评价边界

主代理已核对数据：train719（671双路）、validation228（206双路）、test378（348双路）。使用既有固定ID及类映射；单路样本保留。这里引用既有准备结果，本支线不重复全量数据审计。

| 条件 | CPINet原文 | 当前选择 |
| --- | --- | --- |
| 图像 | 64×64、逐图像素均值/std | native160原尺度裁剪/补零，按footprint有效像素计算均值和总体std；归一化后padding恢复0；不把target mask当统计mask |
| 类别/划分 | OpenSARShip三/五类，5次随机划分，每划分3次取中位数 | 当前OSS2四类、固定训练/验证/测试；种子17/29/43 |
| 增强/平衡 | 翻转、旋转、平移、Gaussian noise，并将每类扩充到相同数量 | 配对D4变换，保持VV/VH空间对应；每batch各类8条、有放回采样；不增加原文平移/噪声 |
| 训练 | 300epochs，batch32，SGD，初始LR0.01，150/200/250轮各降十倍 | 保留300epochs及衰减；seed17比较0.01/0.003，由验证集选择，再运行29/43 |
| checkpoint选择 | 未在已读内容中明确当前这种验证选择流程 | 验证BA优先，其次OA，再其次NLL；当前每10epochs及第1epoch评价，选择范围须按实际执行报告 |
| 预训练/解冻 | 已读内容未确认ImageNet预训练 | 本轮随机初始化、全模型训练；不同于Full的ImageNet起点与局部微调 |
| 缺路 | 原方法为配对双路，没有当前缺路规则 | 四头按availability屏蔽；另报共同双路子集，不能用348与其他方法378直接比较 |

当前每epoch `ceil(719/32)=23` 个平衡batch，即736次样本曝光；300epochs是6900个更新、220800次曝光/运行（D）。这不是每个样本每轮恰好看一次，也不等于原文离线扩充后的相同曝光量。原文300epochs只匹配轮数设定，不应写成与Full或原文完全等算力。

两种LR的seed17均完成、选中者复用，再补29/43时，计划为4个完整训练运行；正式结果是选定LR下3种子。此处是预算规范，不是已完成结果。测试集不参与调参；历史已使用的测试队列不宣称新盲测。每轮正式评价后仍按项目约定回看原始VV/VH chip，保留持续对错及预测翻转案例；本支线没有执行评价或案例分析。

## 10. 独立代码复核结论与最少推进建议

已核查正确：Dense通道递推、五级空间尺寸、IMDFF排除目标自身后残差相加、转置卷积上采样、GBP所有i<=j组对、跨极化门控、FBC的lambda/2与max、单极化自融合输入、四路同一FC、有效头损失/概率屏蔽、GradNorm目标detach及独立权重梯度。

已修正：`Excitation.reduction=2`已在磁盘代码确认；第一epoch等权和L0均值也已确认。重启进程是否加载新版本由主代理负责，本支线不触碰其训练。

未获取/需披露：stem初始通道、TL卷积/压缩、GBP组数及归一化、GBP excitation输入维度补全、转置卷积细节、MS输出宽度、分类embedding选择、FBC参数共享/bias、dropout具体位置、集成规则、GradNorm更新频率/权重优化细节、native160与缺路协议。它们已经给出可执行选择，不再作为等待不可得全文的前置条件。

最少可行推进：以修正C/2后的当前配置重新启动，保留骨干 `10/28/55/91/136` 和现有G8；在协议中写清“依据论文独立实现的SAR-DenseNet型CPINet，含公开FBC组件归一化补全及native160/缺路适配”。保留未归一数值失败与C/16偏差运行，不读取其测试来挑配置。当前未发现另一项必须为了等待文献而阻止正式训练的问题；训练态BN的缺路影响作为明确实现边界保留，不宣称已完全隔离。

本报告不声称完成CPINet性能复现、验证了原论文成绩或已证实跨场景收益。本支线没有启动GPU任务、修改模型/训练代码或正式稿件，也没有提交/推送Git；唯一项目写入为本文件。临时PDF页面渲染仅用于阅读。
