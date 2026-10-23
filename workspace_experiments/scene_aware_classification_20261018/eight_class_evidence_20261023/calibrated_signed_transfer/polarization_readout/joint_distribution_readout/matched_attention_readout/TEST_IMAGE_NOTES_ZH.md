# 历史test四结局实际查看

已实际用view_image查看panels/test_rescued.png、test_harmed.png、test_CC.png、test_WW.png；加源四图，本阶段共8图，主流程亦已查看。取样固定seed17、相对B的四结局首ID，不按图像外观择样。图中full为原additive，attention为本轮双向替换；两者共用冻结B。以下margin一律固定为attention最终最强错误类，不能混用full自己的rival。

|案例|原图可见事实|模型行为与限制|
|---|---|---|
|5030，general/Busan，fold0 epoch75，救回|VV/VH均64×64；VV斜向较宽亮体并伴框外细长亮线，VH主体呈更分离的亮区。|B判bulk0，full和attention均判general2。固定rival0的margin为-.15439、+.69254、+.45691。不是attention独有救回，也不能把VH两亮区命名成某船型物理标志。|
|7310，bulk/Fujairah，fold1 epoch55，误伤|两路72×72，斜向连续长亮体，周围纹理较明显。|B原对，full和attention均错crude5。margin+.02655、-.35564、-.25721。attention减轻负分差但没有修复full错误，不能写成救回。|
|3494，bulk/Busan，fold0 epoch75，CC|两路64×64，横向明亮主体；VV框外背景显示更亮，VH沿主体有较暗缺口。|三者均对。固定general2的margin2.35965、3.01273、2.80313。两种读出都增强本例竞争分差；图中h/b非空间贡献图。|
|3504，bulk/Busan，fold0 epoch75，WW|两路64×64，斜向亮体、周围较强纹理，VH主体下侧亮形不均匀。|三者均错general2，margin-.64017、-.72036、-.74673。两种残差都使该竞争更不利；不能将WW解释为模型没有响应或学不到任何信息。|

四例没有展示两种聚合之间的独有翻转，但源1668/0438已经展示各自不同成功/失败；整体test相对full独有正确75/58/49、独有错误76/69/39，以配对表为总体证据，不能用四图替代计数。

原PNG、160实际crop/pad、valid和记录OBB均可追溯至test_review_cases.csv。双向attention的空间权重不等于物理位置归因，h/b是C4平均的输入摘要，最终非线性头同时使用B和均值。RF混合框内外，不能推断尾迹、二面角、船首尾或像素因果贡献。
