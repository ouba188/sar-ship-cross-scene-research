# DFWA-Net 官方源码可访问性与实现边界审阅

检查日期：2026-10-09（Asia/Shanghai）。仅只读访问公开网页/API/raw 文本，并阅读用户提供的本地 PDF；未下载执行仓库代码、未安装依赖、未运行实验。此次仅新增本文件。

## 结论

**本次未能取得官方源码，因此不能声称完成了逐函数代码审阅，也不能声称仓库里没有实现或没有权重。** 论文首页指向的仓库返回 404；作者账号可访问，但匿名 API 返回的公开仓库列表为空。可能是尚未公开、转私有、删除或迁移，现有证据无法区分。

论文本身可以支持方法层面的借鉴。两个会直接影响原型正确性的事实是：

1. **图1明确包含 iDWT**：小波分支是 DWT -> WSA -> WRA -> iDWT -> DDF。WRA 的跨子带分组不是逆小波变换，不能因正文公式只展开 DWT 而省略 iDWT。
2. **DDF 正文确实写硬二值掩码，但没有交代可导训练路径。** 不能擅自说官方实现用了 softmax、STE 或 Gumbel，也不能把论文写法直接定性为官方代码错误。

## 1. 证据来源与访问结果

论文：Shuaiqi Liu, Wenjing Jiang, Yue Yu, Bing Li, Yudong Zhang, Qi Hu, *DFWA-Net: Dual-Domain Feature-Enhanced With Wavelet Attention Network for SAR Ship Detection*, IEEE Geoscience and Remote Sensing Letters, vol. 22, 2025, article 4011605。

- DOI：https://doi.org/10.1109/LGRS.2025.3601026
- 用户给出的 IEEE 页面：https://ieeexplore.ieee.org/document/11131202
- 实际阅读 PDF：`E:/Docms/格式修改/1008/相关文献/DFWA-Net_Dual-Domain_Feature-Enhanced_With_Wavelet_Attention_Network_for_SAR_Ship_Detection.pdf`
- PDF 第1页印刷链接（换行合并）：https://github.com/Wenjing-Jiang-hbu/DFWA-Net
- 页面图像核对：本目录已有 `paper_page_2.png`（图1、图2）、`paper_page_3.png`（图3、表I、表II），未改写这些文件。

| 实际请求 URL | HTTP/返回内容 | 能说明什么 |
|---|---|---|
| https://github.com/Wenjing-Jiang-hbu/DFWA-Net | 404 | 该地址当前无法匿名读取，不等于确定仓库从未存在 |
| https://api.github.com/repos/Wenjing-Jiang-hbu/DFWA-Net | 404 | 未取得仓库元数据、默认分支、许可证信息 |
| https://api.github.com/repos/Wenjing-Jiang-hbu/DFWA-Net/git/trees/main?recursive=1 | 403，API rate limit exceeded | 此请求受限，不能据此判断目录为空；main 仅为探测分支，未确认其存在 |
| https://raw.githubusercontent.com/Wenjing-Jiang-hbu/DFWA-Net/main/README.md | 404 | 未取得该候选路径文件 |
| https://raw.githubusercontent.com/Wenjing-Jiang-hbu/DFWA-Net/master/README.md | 404 | 未取得另一常见候选路径文件 |
| https://github.com/Wenjing-Jiang-hbu | 200，标题显示 Wenjing-Jiang-hbu (Jevlynne) | 作者账号页面可访问 |
| https://github.com/Wenjing-Jiang-hbu?tab=repositories | 200，未提取到该账号仓库链接 | 页面中未找到可沿用的迁移入口 |
| https://api.github.com/users/Wenjing-Jiang-hbu/repos?per_page=100 | 200，JSON 为 `[]` | 该次匿名查询没有返回公开仓库 |

因此，下文所有模块内容均标明为**论文描述或我们的实现推论**，没有虚构 `models/*.py`、模块类名、行号、配置路径、checkpoint URL 或源码差异。

## 2. WSA / WRA：论文可确认的计算结构

依据 PDF 第2–3页，II-B、图2、式(1)–(5)：

- 输入特征 `X: H x W x C` 经二维 DWT 得到 LL、LH、HL、HH 四个子带。
- 按图2与式(2)排列为四象限：上排 LL/LH，下排 HL/HH，得到与输入空间尺寸相同的拼接布局 Y。
- WSA 在子带内部按局部组计算 Q/K/V 注意力，目的是分别处理不同频率的信息。不是一个对所有子带全局混合的通道注意力。
- WRA 以半高、半宽间隔，从四个子带抽取同源位置的嵌入组成 2 x 2 的四成员组，再计算注意力，目的是跨子带交互。
- 组内计算采用 `softmax(Q K^T / sqrt(d_k)) V`，Q/K/V 的投影是可学习参数。
- 图2用 WSA 的 2 x 2 分组示意；第4页正文与第3页表I给出的最终 WSA 组大小为5。示意图不是最终配置。

**实现尚未确认**：注意力头数、QKV 偏置、位置编码、LN/残差细节、补边和裁切策略、子带重排的确切轴次序，以及 G=5 对各层尺寸的具体处理。

**对当前分类原型的实际影响**：如果直接采用128像素输入/不同特征层，不能照抄检测网络的 G=5。需依据实际子带尺寸选择可整除组大小，或明确补边策略；不能让局部分组跨过子带边界后仍声称是在做独立子带注意力。这是输入尺度转换时会直接遇到的实现问题，不是已确认的官方 bug。

## 3. DWT 与 iDWT：必须保留的区别

依据第2页图1、图2和第3页表II：

- 图1在 WRA 之后明确画出 **IDWT**，之后才进入 DDF 与空间支路融合。
- WRA 在子带表示上建模；iDWT 将处理后的子带还原为与空间支路可对齐的特征图。仅把四象限拼起来恢复 H x W 的张量尺寸，不等于恢复原空间坐标语义。
- 论文比较 Haar、Db2、Db3、Db4，综合精度与效率选择 Haar；表II中 Haar 的整网参数量5.9M、FLOPs 11.8G、FPS 79.4。它们不是独立注意力模块的开销，也不是当前八分类模型的开销。
- 原文将 LH 称为水平高频、HL 称为垂直高频；复现时应按采用的变换库核对轴约定，不能仅凭名称假定其完全一致。

**无法从源码确认**：采用何种 DWT 库、Haar 滤波器归一化、边界模式、是否自定义 autograd，以及 DWT/iDWT 的具体函数调用。本次没有取得任何官方函数实现。

可借鉴的是完整的“分解 -> 子带内/间建模 -> 逆变换 -> 融合”结构。若我们的最小探针只输出子带统计向量用于支持案例匹配，应明确它是 **DFWA 启发的描述符探针**，不是 WSA/WRA/iDWT 的网络复现。

## 4. Gabor 空间支路

依据第2页 II-A 与图1：空间支路使用 Gabor 调制卷积；图中明确为 `Gabor Conv 5 x 5 -> BatchNorm -> ReLU`。原文引用 Gabor Convolutional Networks（2018）作为方法来源。

**不能由现有证据确认的实现细节**：Gabor 方向和尺度数量、频率/带宽/相位设定、Gabor 参数是否可学习、调制发生在卷积核还是激活上、初始化与方向聚合方式。

因此，固定 Gabor 滤波器组可以作为我们自行设计的方向信息探针，但不能写成“直接采用官方 Gabor 空间支路”。尤其固定滤波输入图像与用 Gabor 调制可学习卷积核，不是同一种实现。

## 5. DDF：硬掩码与可导门控的边界

依据第3页 II-C、图3、式(6)，论文流程为：

`concat(Y_s, Y_w) -> adaptive average pooling -> 1x1 convolution -> channel weights -> split(W_s,W_w) -> compare -> masked fusion`

公式是：

```text
mask_s = (W_s > W_w)
mask_w = (W_w > W_s)
Y_f = mask_s * Y_s + mask_w * Y_w
```

正文强调严格二值选择，解释基于单类船舶检测对显著特征与通道域一致性的需要。图3没有画出软门控或梯度估计器；正文称权重归一化，但未给出足以确定激活函数的公式。

**数学/实现推论，不是官方源码结论**：若直接用普通布尔比较生成掩码，而且评分分支只有这一条损失路径，常规反向传播不能通过比较操作训练生成 `W_s/W_w` 的门控参数；被选中的 `Y_s/Y_w` 特征路径仍可以接收梯度。不能误写成“整个模型不能训练”。正文没有说明采用了额外损失、软替代、STE 或其他估计器，故训练方式未解决。

对当前分类原型，若需要学习融合，可明确采用我们定义的软竞争门控：

```text
[a_s, a_w] = softmax([l_s, l_w], domain_axis)
Y_f = a_s * Y_s + a_w * Y_w
```

这是**相对论文硬掩码的主动改动**，不是已确认的官方实现；其意义是保留可导的域选择，并允许细粒度分类同时利用两域证据，不应仅凭这项常见改动宣称原创。是否优于不带门控的双域表示，仍需主线实验回答。本侧线没有实验结果。

## 6. 参数、配置和权重公开情况

| 项目 | 本次核实状态 |
|---|---|
| 整网参数量 | 论文表II报告5.9M；未取得可运行模型核算 |
| 结构尺度 | 图1给出阶段通道64/128/256/512，起始卷积32通道；不是完整配置文件 |
| 训练设置 | 第3页报告640 x 640输入、SGD、初始学习率0.01、1000 iterations、Quadro RTX 8000；保留原文 iterations 口径，不改写成 epochs |
| WSA 组大小 / WRA | 论文最终 WSA G=5；WRA G=2，半尺寸间隔 |
| 小波选择 | 论文采用 Haar |
| 预训练/检测权重 | 未取得官方仓库或下载链接，无法确认是否公开 |
| 许可证 | 未取得，无法确认代码复用授权 |
| 运行入口、环境依赖、训练配置 | 未取得，不能编造 |

## 7. 对主线的可执行建议（不在此处实施）

1. **目前可借鉴算法，不能宣称已复用官方代码。** 可独立实现标准小波分解或使用项目已有可信实现，明确标注为论文启发；本次不安装、不引入第三方仓库。
2. 先分清原型目标：若只检验“高低频信息是否能帮助分类”，轻量子带描述符已可回答初步问题；若要检验论文式注意力，就需要空间特征图、WSA/WRA、iDWT及可学习融合，不能用全局2048维向量替代后称为完整复现。
3. 当前问题是船型多分类与少标注跨港适配；原文验证的是 SSDD/HRSID 检测 AP。原文检测提升不等于我们的 OA/BA 会提升，尤其不能预设高频全是船体细节、低频全是背景。
4. 不把已有四方向平均丢掉的残差直接等同于小波高频。前者是表示对图像旋转的变化，后者是空间尺度/频带分解；能否提供额外类别证据，需要主线另行验证。

**交付边界**：官方源文件、实际硬/软 DDF、预训练权重和许可证均未核实；图1包含 iDWT、正文 DDF 使用硬比较、Gabor 支路结构与两类注意力的分组逻辑已由本地 PDF 核实。没有将访问失败描述成“官方仓库无内容”，没有声称新增分类收益。
