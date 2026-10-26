# 原图回看记录

2026-10-26：以下8张面板均已实际调用view_image查看，**没有未查看面板**。全部为预定anchored_raw_summary、seed17；每图包含query的VV/VH及实际检索的同类2例、混淆类2例VV/VH。沿用旧strong_base_refit/gallery.py的native160裁切/填充流程，无几何缩放、无新特征提取；原图路径、原始尺寸、placement和参考索引见cases.json。

选样：每方向在救回/误伤/CC/WW非空类别中，选bare→final固定真类/竞争类分差变化绝对值最大者；并列按bare相对原rawMLP分差变化排序。正向seed17无翻转，补为2CC/2WW；反向为四类各1例。8例按结果选取，不能估计总体比例。以下“原→bare→final”指原rawMLP、新anchored分数、frequency最终分数；所有分差使用该例同一个竞争类。

## OSS1→OSS2：全部M3零增量

此seed的VV/VH都选零步，849个CC、770个WW，没有救回或误伤。图片所见支持不能解释为M3已利用的证据，因为最终两个残差均为零。

| 已查看图片（gallery/） | 可见事实及真实参考 | 分数与作用阶段 |
|---|---|---|
| oss1_1_CC_2359.png | query为小型近竖直亮条，VH中亮条较连续；chemical参考913/11与bulk508/880均有细长亮区，朝向不同。 | 真chemical对bulk分差31.719→6.428→6.428，仍判chemical；锚定改变了置信差距，M3未改动。 |
| oss1_2_WW_2007.png | query近竖直，VV上端更亮，VH有沿长轴的多个亮点；container参考798/764近横向，general1220较短且竖直、1151斜向。 | 真container对general分差9.135→-22.618→-22.618；原rawMLP正确，新bare已转为general，M3没有修复。 |
| oss1_3_CC_1744.png | query VV有短横向亮区，VH框内较弱，框外左上角两极化均有亮点；chemical474/759与general448/479的短条结构有重叠。 | chemical对general分差-1.440→18.907→18.907；原预测bulk，新bare/final均chemical。变化发生在bare，不能归给M3或框外亮点。 |
| oss1_4_WW_2208.png | query近横向、内部多个亮点，native占幅大于多数参考；container798也有多个亮点，174较短；general1220/159较短。 | container对general分差10.770→-18.890→-18.890；原预测bulk，新预测general，始终未判对。原正分差只针对general，不代表原四类预测正确。 |

## OSS2→OSS1：VV读出同时救回与误伤

该seed共有39救回、43误伤、513CC、326WW。源选择alpha=1、frequency100步。JointHead是独立比较臂，不参与下面的最终融合。

| 已查看图片（gallery/） | 可见事实及真实参考 | 分数与作用阶段 |
|---|---|---|
| oss2_1_rescue_1302.png | query为短斜条，VV离散亮点、VH较连续；bulk255/2517和container1531/560均呈细长亮区，没有肉眼唯一判据。 | bulk对container分差0.259→-3.318→0.199。VV对两类残差分别+0.569/-3.190，固定B已救回至+0.441；frequency权重0.931/0.069使分差回落至+0.199，救回来自VV读出而非gate新增。 |
| oss2_2_harm_1144.png | query为小竖条，VV中心较亮，VH可见数个亮点；general274/1597为较短斜/横条，container1985更长且多亮点，249较短。 | general对container分差-5.033→2.353→-0.575。VV对general为-3.382、container为-0.326，固定B已误伤至-0.704；gate权重0.959/0.041稍缓和但未救回。 |
| oss2_3_CC_58.png | query为横向长条、VV/VH均有多个亮点；container1985也有横向多亮点，2017为竖向；chemical2571/1914更短。 | container对chemical分差19.035→100.647→54.171。VV将真类压低46.172、竞争类提高13.994，固定B分差40.481；gate减少VV比重至0.758，将分差恢复至54.171，仍正确。不能把大幅降置信当作分类退化或物理解释。 |
| oss2_4_WW_404.png | query为很小的斜亮条，两极化均只占少量像素；general274/1597和chemical2571/2716均有短亮区，chemical2716也为斜向。 | general对chemical分差-4.469→-18.092→-6.187。VV对general+2.520、chemical-9.385，改善11.905但不足翻转；gate权重1/0，无额外改变。 |

## 同8图的四类分差对照

下表为bare减原rawMLP的**中心化四类logit**变化，顺序bulk/general/chemical/container。该对照不增加图片；各类真类相对分差及完整前后分数均保存在cases.json。

| query | 四类中心化变化 |
|---|---|
| OSS2 2359 | +17.467 / -12.302 / -7.823 / +2.658 |
| OSS2 2007 | -11.321 / +19.740 / +3.594 / -12.013 |
| OSS2 1744 | -7.744 / -5.087 / +15.260 / -2.428 |
| OSS2 2208 | -8.610 / +18.575 / +1.121 / -11.086 |
| OSS1 1302 | -1.860 / -1.421 / +1.564 / +1.717 |
| OSS1 1144 | +2.233 / +4.815 / -4.478 / -2.570 |
| OSS1 58 | -37.006 / -8.856 / -17.875 / +63.737 |
| OSS1 404 | -0.247 / -4.726 / +8.898 / -3.925 |

可保留的证据是：bare改变四类相对分数可以同时产生救回与损失；M3零步时不会自动修复bare的错误；反向非零VV读出能抑制某些错误竞争类，也会压低原本正确的真类，frequency仅部分缓和。这些作用阶段由实际残差和权重支持。图像显示短条、多亮点及占幅在同类和异类参考间有重叠，但尚未证明具体像素机制或物理因果；本轮不追加干预、重训或扩图。

全部exec session已结束。8/8面板已实际查看，gallery_receipt.json同步更新为visual_review_pending=false。
