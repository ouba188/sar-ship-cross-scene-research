# 原始 VV/VH 案例查看记录

本记录对应最终 private/OSS2 seeds 17/29/43。最终画廊含 23 个比较案例、20 个不同查询、46 个 VV/VH 面板。20 个查询的 VV/VH 及实际同类/混淆类 reference 图像均已查看；相同 query/ref 组合在多个比较中的重复展示复用已有查看。早期 seed17 另外查看的案例单列在末尾，不冒充最终选图。

原图直接读取 manifest 指定的 8-bit display chip，保留原像素；每个 tile 分别展示原图、160 像素精确 crop/pad、独立 OBB 叠加。没有 resize、重新拉伸或将显示亮度当作物理后向散射。VV/VH 分开。每个比较文件夹的 `native_sources.json` 保存原始绝对路径、raw_size、placement、实际可用状态、reference ID；`cases.json` 保存所有三个 seed 的 logits、预测和固定 rival 分差。

选择规则：每状态一个不同 query，按该状态出现顺序数、固定 rival 分差变化绝对值、ID、seed 排序；无随机抽样。同类与 rival references 为实际固定 bank 中排除同 ID/MMSI 后的前两条可用记录，不是 attention 排名。结果选择的代表案例不能估计总体比例。自动生成的 JSON/HTML `not_reviewed` 是渲染初始状态，**实际人工查看范围以本记录为准**；没有让脚本自动宣称视觉审阅完成。

## 最终选图路径

下列路径均相对本 `analysis/` 目录；每行 ID 的两个文件名为 `{状态}_{完整ID}_vv.png` 与 `{状态}_{完整ID}_vh.png`，链接入口列出原尺寸图片和完整元数据。

| 比较目录与入口 | 最终选图，括号内为选择 seed |
|---|---|
| [private/adapt_only_vs_saved_baseline/index.html](private/adapt_only_vs_saved_baseline/index.html) | rescued_32638 (17), harmed_6194 (43), CC_22789 (17), WW_28198 (29) |
| [private/source_mix_vs_saved_baseline/index.html](private/source_mix_vs_saved_baseline/index.html) | rescued_31926 (17), harmed_6194 (43), CC_22789 (17), WW_28198 (29) |
| [private/source_mix_vs_adapt_only/index.html](private/source_mix_vs_adapt_only/index.html) | rescued_5125 (17), CC_11160 (29), WW_31141 (17)；三顺序均无 harmed |
| [oss2/adapt_only_vs_saved_baseline/index.html](oss2/adapt_only_vs_saved_baseline/index.html) | rescued_pub2_opensarship2_001098 (17), harmed_pub2_opensarship2_003355 (17), CC_pub2_opensarship2_000960 (29), WW_pub2_opensarship2_001045 (29) |
| [oss2/source_mix_vs_saved_baseline/index.html](oss2/source_mix_vs_saved_baseline/index.html) | rescued_pub2_opensarship2_002877 (43), harmed_pub2_opensarship2_003678 (43), CC_pub2_opensarship2_000955 (43), WW_pub2_opensarship2_000932 (17) |
| [oss2/source_mix_vs_adapt_only/index.html](oss2/source_mix_vs_adapt_only/index.html) | rescued_pub2_opensarship2_000922 (43), harmed_pub2_opensarship2_001089 (43), CC_pub2_opensarship2_001056 (43), WW_pub2_opensarship2_001072 (43) |

例如公共增量救回的准确文件为 [VV](oss2/source_mix_vs_adapt_only/rescued_pub2_opensarship2_000922_vv.png) / [VH](oss2/source_mix_vs_adapt_only/rescued_pub2_opensarship2_000922_vh.png)，对应误伤为 [VV](oss2/source_mix_vs_adapt_only/harmed_pub2_opensarship2_001089_vv.png) / [VH](oss2/source_mix_vs_adapt_only/harmed_pub2_opensarship2_001089_vh.png)。旧 seed17 运行留下的额外 PNG 不属于最终选图，以当前 `selection.json`、`cases.json` 和本表为准。

## 私有实际图像与决策

以下分差均为真类 logit 减固定 rival logit。列中 B/A/M 分别表示 saved_baseline/adapt_only/source_mix，所列为选择 seed 的证据；其他 seed 全部在 cases.json，不能由一个 seed 推断三 seed 都翻转。

| ID / 状态 / seed | 原始 VV/VH 与实际 refs 的可见事实 | 决策证据与边界 |
|---|---|---|
| 5125 / 增量 rescued / 17 | VV/VH 均为 64×64，斜向条状亮区，VV 中可见更长的交叉延伸。bulk refs 34478/31512 与 chemical refs 24527/7790 都包含紧凑或细长亮区，不能据形状直接定类。 | bulk-chemical B/A/M=+0.4310/-0.0737/+0.0855。仅 seed17 恢复原模型已正确的判断；seed29 两臂仍错、seed43 两臂均对。不是新增修复原模型错误的例子。 |
| 11160 / CC / 29 | VV/VH 64×64 的小紧凑亮区。offshore refs 13000/13276 同样有小亮区；general refs 6913/22755 的形状与背景各异，VH 仅 6913 可用。 | offshore-general A/M=4.7307/4.1136；三 seed 均正确，但 mix 的这一分差均小于 control。持续正确不能证明证据变强。 |
| 31141 / WW / 17 | VV 72×72、VH 80×80，query 为较宽横向条状亮区；offshore refs 13000/13276 小而紧凑，crude refs 29922/2077 则更长。 | offshore-crude A/M=-8.3160/-9.0781；三 seed 都错，seed43 top1 为 bulk 而非固定 rival crude。外观差异仅提出支持覆盖假设，不证明误判由它造成。 |
| 31926 / 相对 baseline rescued / 17 | VV 88×88、VH 96×96，近竖直细长亮区。同类 crude refs 29922/2077 是较横向/斜向条状，bulk refs 34478/31512 也含细长结构。 | crude-bulk B/A/M=-0.3823/+0.0593/+0.1151，两种续训均救回。真类分数也下降、bulk 下降更多（主代理同样核对），因此救回不是“真类绝对响应增强”的证据。 |
| 32638 / adapt rescued / 17 | VV 64×64 可见明显交叉延伸和右边缘条带；VH 为紧凑小亮区、背景纹理较显著。同类 chemical refs 24527/7790 与 bulk refs 34478/31512 都有局部形状重叠。 | chemical-bulk B/A/M=-0.2780/+0.3322/+0.1840；两臂均救回，mix 分差更小。不能命名交叉结构的物理原因。 |
| 6194 / harmed / 43 | VV 64×64 有较长交叉延伸，VH 是更紧凑的竖向细长亮区。同类 chemical 24527/7790 与 tug 12041/7429 都包含紧凑或细长例子。 | chemical-tug B/A/M=+0.3505/-0.2247/-0.1668，两臂仍误伤；seed29 实际错为 general cargo，不能用 seed43 的固定 rival 解释全部顺序。主代理亦查看本例 VV/VH。 |
| 22789 / CC / 17 | VV/VH 64×64 中心小、近竖向亮区；offshore 两个 refs 较紧凑，bulk refs 更长且部分有交叉延伸。 | offshore-bulk B/A/M=1.4990/3.3153/2.7294；三 seed 都对，mix 不及 adapt 的分差增幅。形状相符不是模块机制证明。 |
| 28198 / WW / 29 | VV 88×88 有长交叉延伸及右上局部亮结构；VH 80×80 更紧凑。offshore refs 很小，而 bulk 34478 也有交叉延伸。 | offshore-bulk B/A/M=-16.7887/-14.7794/-15.4151，仍远离正确决策；17/29 从错误 bulk 转为错误 crude，43 仍为 bulk。分差改善与错误类别变化不等于 rescue。 |

私有增量比较不存在 harmed；该缺组已写 selection.json，未拿相对原模型的误伤充当增量误伤。原模型比较中的损失仍保留：chemical 的净损失多于净救回，整体 OA 下降。

## OSS2 实际图像与决策

下表 ID 简写为末六位，完整 ID 为 `pub2_opensarship2_` 加该后缀。实际参考组：bulk=006462/006477，general=004017/005079，product_chemical=006746/005117，container=005219/001541。均为固定 bank，原图尺寸不相同且未统一 resize；例如 container 005219 为 141×141、001541 为 97×97，bulk 006462 为 85×85、006477 为 81×81。

| ID / 状态 / seed | 原图可见事实 | 模型证据与局限 |
|---|---|---|
| 000922 / 增量 rescued / 43 | VV/VH 101×101；近竖向斜条、数个较亮局部，VH 可见较连续的细长轮廓。container 与 bulk refs 都含条状/多亮点结构，姿态并不一致。 | container-bulk B/A/M=-0.2681/-0.2093/+0.0746；mix 在该 seed 修复 baseline/control 均错的查询；17/29 两臂本已正确。 |
| 001089 / 增量 harmed / 43 | VV/VH 73×73；近竖向斜条和数个亮点，与 000922 的部分外观相近，但真类是 bulk；两类参考均有细长或局部亮点。 | bulk-container B/A/M=+0.0855/+0.0703/-0.2428。mix 把 baseline/control 正确样本推向 container；救回和误伤同时体现竞争边界变化，不能把类似条状外观当可靠类别判据。 |
| 001056 / CC / 43 | VV/VH 97×97，较长斜向条带，沿长度可见多个亮局部。两类 refs 的长度、姿态和亮点排列有差异。 | container-bulk A/M=5.3833/5.7767，三 seed 均对。分差增加只说明已有正确决策增强，未增加正确样本。 |
| 001072 / WW / 43 | VV/VH 65×65，较短的近竖向斜亮区，VH 中亮区更集中；与 container refs 的部分局部形状重叠。 | bulk-container A/M=-1.7861/-2.1111，三 seed 持续错判。不能用原图外观证明标签错误或成像机制。 |
| 002877 / 相对 baseline rescued / 43 | VV/VH 49×49，短斜亮区、一端较亮；chemical refs 006746/005117 有相近的斜向多亮点，bulk refs 也有局部亮结构。 | chemical-bulk B/A/M=-0.5170/-0.1325/+0.1354。mix 跨过边界，control 未跨过；这是实际局部新增收益，但不足抵消其他样本损失。 |
| 003678 / 相对 baseline harmed / 43 | VV/VH 89×89，近竖直细长亮区；与实际 bulk 和 chemical refs 的姿态不同，两类都具有细长响应。 | bulk-chemical B/A/M=+0.2529/-0.1733/-0.4435，两臂误伤，mix 更负。仅凭形状不能解释该次偏置改变为何误伤。 |
| 000955 / CC / 43 | VV/VH 69×69，近竖直细长亮区；与上一例具有一些相似视觉元素，仍存在亮度分布、大小差异。 | bulk-chemical B/A/M=3.5258/3.0878/2.8320，三 seed 均对。类似向 chemical 的竞争变化在较大初始分差上不翻转，不能由最终对错反推所见形状一定有用。 |
| 000932 / WW / 17 | VV/VH 81×81，略弯的近竖直细长亮区，局部亮度不均；container 005219 较竖向且多亮点，bulk refs 较横向，但不能据姿态定类。 | bulk-container B/A/M=-0.3068/-0.9964/-1.0211，三 seed 持续错。mix 没修复这类已有错误。 |
| 001098 / adapt rescued / 17 | VV/VH 53×53，短斜条和数个亮局部；general refs 004017/005079 也有短亮区，bulk 006477 同样是条状。 | general-bulk B/A/M=-0.5230/+0.2616/+0.0385，两臂均救回、mix 更靠边界；29/43 两臂仍错。 |
| 003355 / adapt harmed / 17 | VV/VH 53×53，横向紧凑短亮区；chemical 与 general refs 的局部紧凑/条状形态均有重叠。 | chemical-general B/A/M=+0.3079/-0.5258/-0.3418，两臂都错；mix 缓和分差却未救回，其他顺序也错误。 |
| 000960 / CC / 29 | VV/VH 45×45，短斜向局部亮区；general 004017/005079 是不同姿态的紧凑亮区，bulk refs 更大但原尺寸展示。 | general-bulk B/A/M=5.1390/6.0770/5.8153，三 seed 均对；mix 增强幅度不及 adapt。 |
| 001045 / WW / 29 | VV/VH 77×77，接近水平的两个较亮局部及横向连接；general 与 bulk refs 中都有离散亮点/条状元素。 | general-bulk B/A/M=-4.0827/-3.1763/-3.4239，三 seed 都错；边距靠近零不等于改变识别结果。 |

公共最终 12 个查询的 VV/VH 与实际 references 已查看；早期已查看且最终仍保留的 001098、003355、000932 复用原图查看。最终新增的 000922、001089、001056、001072、002877、003678、000955、000960、001045 均另行查看其两个极化。

## 先导案例与结论边界

seed17 先导阶段另看过私有 24863、23405 和公共 003580、003610、003325、003502、001113。24863 是小而紧凑的 chemical query，两边 refs 都有紧凑/细长形态，mix 缓和分差但仍误伤；003580 是保住 baseline 已对的 chemical 样本，而 003610 失去 control 对 baseline 的 container 救回。这些先导事实保留，不计为最终图像选择中的额外独立证据。

可保留的实际收益条件是少数接近类别边界的查询在某些顺序被推向真类；实际损失也出现在相似条件。私有 5125 的收益只发生在 seed17，公共增量净误伤多于救回，不能宣称问题解决。参考外观的跨类重叠和同类差异是真实可见事实，但没有逐参考替换/贡献证据，不能断言某 reference 导致某次分类。

结合主代理 [共同偏移诊断](../diagnostic_summary/SHARED_SHIFT.md)，本轮 centered-logit delta 的 92.4%–97.1% 能量为 query-common offset；OSS2 source_mix seed43 的共同偏移重建复现全部 top1。这使“整体类别竞争偏置改变”比“已学会查询特定的原图关系”更有直接证据。仍有查询特定残差和私有不一致，且该诊断是训练 delta、开发预测均值的事后分析，不是整个 M3 性质、物理解释、分类因果比例或部署方法。**原图展示只支持上述可见事实，研究目标未达成。**
