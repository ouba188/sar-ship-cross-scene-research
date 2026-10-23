# 源四结局原图实际查看

已实际用view_image查看panels/source_rescued.png、source_harmed.png、source_CC.png、source_WW.png。固定seed17，每结局按instance_id/fold首例，相对B定义。所有四例两极化有效；原PNG及实际160 crop/pad同时显示，绿色OBB来自旧记录。面板右下是输入h/b，不是attention位置权重或物理贡献。

|结局/对象|图像可见事实|实际模型证据与边界|
|---|---|---|
|救回 returned_20260915_0001668，bulk，Port Klang，fold1 epoch55|VV/VH均64×64、斜向长亮主体，主体显示白亮区较连续；VH上端形状与VV有差异，周围纹理较暗。|B判offshore7、full仍7，attention变bulk0。固定rival7分差-.50818→+.16095，full为-.14265。不同极化h/b相对真/竞争类的顺序不同，但非线性head同时读B/均值，不能把救回单归某一条纹或某个极化统计。|
|误伤 returned_20260915_0000438，bulk，Jebel Ali，fold1 epoch55|VV/VH均72×72，斜向长亮体，右上端有不均匀亮形，背景纹理比1668更明显。|B及full均正确bulk0，attention判crude5；固定rival5分差+.13632→-.10587，full+.12792。竞争类两pol的h/b均较真类高，但此相关数值不是错误的独立因果归因。|
|CC pair_0032734，bulk，Ningbo-Zhoushan，fold2 epoch0|VV/VH均72×72，近竖直连续亮主体，VH主体显得较窄，周围较暗。|attention和full均0步，分差对general2为2.08997，完全等于B。右下h/b非零不能说明本头使用了它们。|
|WW pair_0032903，bulk，Ningbo-Zhoushan，fold2 epoch0|VV/VH均64×64，近竖直偏斜的细长主体；VV亮区较暗/分段，VH上部亮点更突出。|两模型均0步、保留B错误general2，固定分差-1.39142。不能说attention从图中学到错误机制，也不能由外观断言标签错误。|

四例提供可复核的行为差异，不代表类别总体比例。深层RF混合船体、框外及padding，即使footprint排除补零聚合格，也不成为像素因果遮挡。未命名尾迹、二面角或物理散射正负。
