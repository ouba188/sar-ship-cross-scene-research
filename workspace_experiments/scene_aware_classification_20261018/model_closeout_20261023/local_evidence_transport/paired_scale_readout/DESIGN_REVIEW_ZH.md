# 查询/参考一致的尺度读出：独立设计审查

2026-10-24。范围：审查既有实现与本轮四臂协议；只写本文，不改主线代码、不提取新特征、不训练、不发送其他线程消息。本审查没有运行候选实验，也不把建议写成已完成验证。

## 判断

**可以按此方案开展配套训练。** 既有原图探针说明统一删去细尺度会同时救回和误伤，查询与参考采用一致尺度表示有可检验的收益路径。四臂能区分原始重复输入、多尺度表示、相邻尺度差分及固定随机换基的作用。当前真正要落实的是未 unit 坐标、原生图滤波边界、固定参考索引、零初始化含义和相同训练/选点机会；没有发现原始参考选择使用 evaluation 标签的问题。

本轮应称“冻结主干下的尺度表示与局部读出训练”，不称新增频带信息、严格频率分解、物理机制确认或端到端骨干训练。difference 若胜出，首先支持特定尺度对比与匹配读出的有效性。

## 已核实的参考与数据协议

依据 `../match_readout.py:60`、`:134`、`:153`、`:165` 及现有缓存，原始流程是：仅 fit_train 的有效 token 拟合 PCA；fit_train 内先按 MMSI 去重，再按已知源端类别建立每类 8 张 bank；用查询的原始主体描述与每类 bank 的相似度选择该类 2 张参考。所有 8 个候选类都会计算，查询 class_id 不参与相似度或选参考。参考的已知类别用于组织支持库是合法监督，不等于读取查询真值。

本次只读检查 `upstream/table.csv`、`indices.npz`、`projection_layer1.npz` 和 `scores_layer1.npz` 得到：

| 项目 | 实际结果 |
|---|---|
| 原表 | 3387 条；strategy_fit 1646，utility_train 644，selection 612，evaluation 485 |
| PCA/建库 fit_train | 1205 条，属于 strategy_fit |
| bank | 64 个不同原表索引，全部属于 fit_train |
| 本轮所需并集 | bank64 + utility644 + selection612 + evaluation485 = 1805 条 |
| 参考覆盖 | 这 1805 条的旧 refs 全部落在该 64 张 bank 中 |
| bank 与三个读出队列的交叠 | instance_id、MMSI、pass_group 各为 0 |
| 已有 PCA | components 为 32 x 256；本轮直接复用 |

必须直接复用 `scores_layer1.npz['refs']`，四臂、四通道及三配置均固定同一对参考。不要按分频后的相似度重新挑选，也不要用真类或真实混淆类给推理选参考。真类/混淆类信息只用于训练损失、授权的 source selection 或冻结后的诊断。

1805 条是计算并集，不是训练集。bank 只供参考，644 条才进入残差头监督与输入标准化；612 条只选 checkpoint；485 条在所有候选与 checkpoint 冻结后报告。把原表索引映射到子集局部索引时，refs、mass、raw token、base logits 必须使用同一映射。最小实现可保存 `global_ids` 与一次构造的 global-to-local 映射；不需要重建 bank 或另造身份系统。

## 必须落实的表示与预处理

### 1. 提取未 unit 坐标，再组合，再 unit

令 F_s 是同一原生视图在尺度 s 下，经冻结 R50 layer1、同一组 20 slots 和 C4 平均得到的 256 维描述；s 为 0、1、4、12。P、mu 为既有 PCA 的 components、mean：

```text
Y_s = (F_s - mu) P^T
multi = [Y0, Y1, Y4, Y12]
diff  = [Y0-Y1, Y1-Y4, Y4-Y12, Y12]
repeat = [Y0, Y0, Y0, Y0]
mix_a = sum_s Q[a,s] Y_s
X_a = Z_a / max(||Z_a||_2, 1e-8)
```

`../image_frequency_probe/run.py:115` 的旧 Extractor 已执行 unit，**不能直接拿其返回值计算本轮 difference/mix**。新实现应返回未归一化 F 或 Y；raw 直接从 `tokens.npz['layer1']` 取，不能从已经 unit 的缓存反推。保留原顺序：先 C4 对应 mask 的槽位读出并平均，再投影、组合、unit，不在每个旋转中先 unit 后平均。

三个差分方向固定为细尺度表示减较粗尺度表示，保留正负值。不取绝对值、不 ReLU、不裁到 [0,1]、不按样本翻符号或重新拉伸。相邻差分中 PCA 均值自动抵消；可等价写成 `(F_a-F_b)P^T`，不能再减一次 mu。mix 也应直接作用于四个已经中心化的 Y，不能混合 F 后只减一次 mu，除非显式处理 Q 的行和。

缺失极化和 mass=0 的槽位保持原有无效状态；中心化后的零填充不能成为新有效 token。沿用原 `unit` 的范数下限及原 mass，不临时引入按结果调节的能量阈值。差分接近零时 unit 可能放大很小响应，这是本表示的真实边界；利用已经生成的 utility 数据记录差分范数的量级和近零比例即可。若出现异常收益/退化且集中于近零槽位，再据此纠错，不预先扩展成新一轮门控搜索。

### 2. 原生 PNG 上各自滤波，之后原 placement 补零

沿用 `../image_frequency_probe/run.py:28-43`：PNG 转灰度浮点 [0,1]，在完整原生 PNG 上分别计算 sigma=1、4、12 的 Gaussian，使用原探针的 reflect 边界与同一核截断设置，然后放入原有 160 x 160 placement。三个 Gaussian 都从同一个 raw 计算；不要以 G1 再做 sigma=4、以 G4 再做 sigma=12，否则实际平滑尺度改变。

不要对 uint8 结果反复滤波/量化，不新增直方图拉伸或重新检测 OBB。全部尺度使用原始 geometry、footprint、mass 和相同 C4 masks；只改变图像响应。原补零在输入像素域仍为 0，随后正常执行原 ImageNet mean/std 归一化，不能把归一化后的 padding 强行置 0。查询和全部 64 张参考均走相同处理。

**本项目实际边界：** 对本轮 1805 条读取既有 `tokens.npz['valid']` 和 `view_snapshot.json` 的原图尺寸，得到 3399 个有效极化视图：utility 1205、selection 1173、evaluation 901、bank 120。3399 个全部需要 padding，没有任何一个原生宽或高超过 160，故本轮没有“滤波访问原 160 crop 外额外像素”的范围扩大问题。先 pad 再 Gaussian 却会让人工零边界进入滤波，并将非零响应扩到原 padding，属于本轮真实可达的实现错误。

reflect 是原 PNG 边界的延拓假设，不是真实更大场景。既有探针对 reflect/nearest 的相近结果不能替代本轮训练结果；保留这一限制即可，不要求先重跑边界实验。sigma 单位是 PNG 像素，不能声称各产品共享同一米尺度或对应固定散射结构。

### 3. 可逆性成立的位置必须写准确

归一化前，difference 是四个 Y 的可逆线性重参数化：

```text
Y12 = D3
Y4  = D2 + D3
Y1  = D1 + D2 + D3
Y0  = D0 + D1 + D2 + D3
```

所以 difference 相对 multiscale 没有增加信息。每通道分别 unit 会丢弃尺度幅度，随后每通道独立 OT 又是非线性且有损的读出；最终分数之间没有上述可逆保证。相邻差分仍可能改变可利用的类别相似关系，值得按计划训练。

深网络的 `F(I)-F(G_sigma I)` 不等于 `F(I-G_sigma I)`；这里不能叫“网络编码了输入 DoG 频带”。region 的响应还跨越 layer1 感受野，Gaussian 会把框内亮响应传向邻域。已有 `../image_frequency_probe/FINDINGS_ZH.md` 的来源分解正说明这一点。本四臂没有分离框内/框外输入来源，因此可以测试尺度读出收益，但无法确认纯背景、海面或尾迹通道。无需为启动本轮再加来源分解训练臂。

### 4. mix 是固定尺度轴对照

seed1024 只生成一次 4 x 4 正交矩阵 Q，可用标准正态矩阵 QR 后固定分解约定，保存实际 Q 及乘法方向。Q 只混四尺度，同一个 Q 用于所有样本、VV/VH、槽位、查询和参考；不混 PCA 维度，不逐配置重抽，也不按指标挑 Q。

Q 在 unit 前保留四尺度联合坐标的总平方范数；逐通道 unit 和独立匹配并不保持换基不变性。单个随机 Q 是有用的指定换基对照，不能证明差分优于所有可逆变换。共同把某一输出通道的查询和参考同时乘 -1 不改变该通道余弦相似度，因此不要把 QR 列/行符号约定本身写成性能机制。

## 同容量和训练协议

每通道沿用旧 independent 的 2 极化 x 3 区域 x 2 统计量，先对同类两参考取平均，得到 12 分数；四通道共 48。两个统计量仍是 OT 相似度加权和与保留匹配质量，沿用原 dustbin、epsilon=0.1、60 步和真实匹配质量不重新归一化的规则。不要提前平均四通道或把两参考保留成额外特征。

追加的三个上下文必须沿用 `../train_readout.py:36-49`：当前类中心化 full score、其余类中最大 full score、两者之差。不是另外拼 base/body/full 三种 logits。类别维共享同一个 51->12->1 GELU 头，共 **637 个名义可训练参数**；类间 delta 去均值，最后加在冻结 `strategies_seed{seed}.npz['logits'][:,2]` 上。四臂都从 full 起步，不先叠旧 independent 再加新头。旧 strong independent 是独立保留的对照，M1/M2 链保持冻结。

“零初始化”明确为**只清零最后一层权重与偏置**，第一层按相同 seed 正常初始化，参照 `../train_readout.py:24-33`。若全部层清零，在该共享头和类间中心化下输入无差异，无法正常学到分类残差。每个配置在每个臂初始化前重设相同 seed，或复用同一初始 state，避免循环顺序改变初始参数。

四臂沿用以下完整日程，而不只匹配 epoch 数：

- seed/config 17、29、43 分别对应既有 full logits；同 seed 的四臂成对比较。
- 只在 utility644 拟合输入 mean/std/active，并训练；每个臂按同一规则拟合自己的 scaler，不用全1805条统计量。沿用 std>1e-6 的 active 规则，不按 evaluation 调阈值。
- AdamW，lr=0.003，weight_decay=0.01，200 个 full-batch 步；joint_loss 为普通 CE 与类均衡 CE 各半。
- selection612 上每5步计算 exact J=(OA+BA)/2，包含 step0；仅严格大于旧最佳才更新，平局保留较早点。保留 checkpoint0 和选点曲线，不能强制每臂选到非零步。
- 所有臂/配置的选择结果冻结后再读 evaluation485 指标，不按其结果换 sigma、Q、参考、损失、初始化或选点规则。若要宣布一个最终臂，规则也须依据 source selection 预先明确；四臂在 evaluation 的排序可以描述，不能当作独立确认的胜者选择。

637 参数保证四个新头的架构容量相同，但 **repeat 不具有与其他三臂相同的输入有效维数**。重复的四份原始分数使第一层对应权重可合并为一组；其功能仍是单尺度读出，重复参数会改变优化及正则化。旧 independent 的 15->12->1 是 205 参数，不是本轮同参数量对照。因此 repeat 是合理的同架构冗余输入对照，multi/mix 则帮助判断差分结构；不要要求 repeat 训练后逐点复现旧 independent，也不要仅凭 diff>repeat 宣称带分解特有收益。

## PCA32 的边界及结果判读

既有 PCA 只对 raw 源端描述拟合，低通后有用的变化可能落在被丢弃的 224 维内。本轮共用冻结 P 是清楚且合理的约束，避免把投影变化混入四臂比较；不需要临时为每臂另拟 PCA。负结果只否定本投影/主干/槽位/参考/训练日程下的候选，不能推断图像尺度方向无效。正结果也只是冻结投影中的收益，不证明找到了物理频带。仅当已有新特征和结果显示投影明显限制该路径，再考虑另轮 source-only 投影对照。

| 实际结果 | 本轮可支持的判断与下一步 |
|---|---|
| multi、diff、mix 都比 repeat 好 | 多尺度输入有用；差分特异性未建立，保留实际最稳的 source-selected 方案 |
| diff 比 multi 和指定 mix 好，且超过旧 strong independent | 支持相邻尺度差分在该归一化/匹配读出中的优势；继续定位哪些样本及类别竞争被改善 |
| diff 仅胜 repeat，未胜 multi/mix | 不能将收益归于频带差分本身 |
| 新臂胜 full，未胜旧 strong independent | 新支路有局部收益，但尚未达到已有强局部基线，不替换现有方案 |
| 全部选到0或没有完整收益 | 记录当前约束下无增益，先看训练/选点与实际误差来源，不自动加尺度或换 Q 搜索 |

selection 已被既有多轮探针使用，evaluation485 也已有历史实验读出。无标签泄漏的实现不等于全新独立确认；报告应称既有开发评价队列上的配对比较。三配置不是三套独立舰船样本，也不能把四尺度、四臂或多次曝光当样本量增加。

## 主代理所需最小验证与收尾

只做能改变实施决策的验证，不另起全量审计：

1. **子集映射与 raw 重放。** 在缓存映射后确认 instance_id/refs 对齐；repeat 的每份局部分数应与旧 independent 缓存一致到已有数值误差量级。失败则先修索引、mask、C4 或 OT，不训练掩盖偏差。无需要求新 637 参数头训练结果复现旧 205 参数头。
2. **组合顺序。** 用生成的少量有效 Y 检查 diff 累加还原、Q 的正交性/逆变换、同一 Q 的查询参考应用及无效槽位归零。失败则修“先 unit 后差分”或矩阵轴错误。检查到 unit 前即可，不要求 unit 后仍可逆。
3. **预处理 spot check。** 一张实际需 padding 的有效 PNG，确认 G1/G4/G12 来自原生浮点图、放回 placement 后 padding 保持原值且 mask 不变。失败则修滤波位置/量化；不重跑上一轮全部边界干预。
4. **初始输出和参数。** 四臂名义参数均637，step0 delta=0、输出与 full 一致、同 seed 初始 state 一致。失败则修初始化或 base 连接。

本侧线已完成上文的角色、bank、并集与图像尺寸只读核对；以上新候选实现检查由主代理随实现执行，尚未执行。

实验结束后无论涨点、持平、退化，沿用项目要求回看持续正确、持续错误、救回、误伤原始 chip。VV/VH 分开，展示实际同类及混淆类两张参考，并把原图、各尺度输入、局部分数和最终分差对应起来；没有翻转时仍解释证据变化。优先看四臂分歧案例，记录路径、ID、配置与抽样依据，不把亮纹命名为已证实的散射机制。结果由主代理按本轮范围统一归档和 GitHub 同步，本侧线不另行推送或修改 brain。

## 审查依据路径

以下相对路径均以本文所在 `paired_scale_readout/` 为基准：

- `../image_frequency_probe/FINDINGS_ZH.md`：前轮原图证据、边界对照、来源分解与结论范围。
- `../image_frequency_probe/run.py:28-43,91-119,151-167`：原生滤波/placement、旧 Extractor 提前 unit、原参考读取。
- `../extract_tokens.py:29-40,93-112`：20 slots 的 body/side/end 顺序与 C4 读出。
- `../match_readout.py:26-57,60-78,134-167,181-189`：unit、OT、建库、PCA、固定两参考及参考平均。
- `../train_readout.py:24-49,69-115`：头、三个上下文、训练日程和 exact J checkpoint 选择。
- `../../regional_utility_pilot/upstream/pipeline.py:39-46,67-75`：上游 fit 角色、J 和 joint_loss 定义。
- `../../../eight_class_evidence_20261023/obb_dynamic_readout/common.py:92-104`：原生视图和 placement 接口。
- 工作区 `same_vessel_gallery_20261019/local_condition_module/native_input.py:6-37`：不缩放 crop/pad。

本文的可逆性与读出边界来自上述明确公式和实际代码；未增加外部文献的经验性断言。四臂已经足以回答本轮问题，无需等待长综述或新增训练臂。
