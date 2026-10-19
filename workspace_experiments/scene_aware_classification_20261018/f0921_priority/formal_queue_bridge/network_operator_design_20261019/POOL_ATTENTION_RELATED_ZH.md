# P/A池化与注意力：三篇近邻及数学边界

2026-10-19。结论：P可定位为保留区域边际、显式组织相对位置跨区共现的任务化池化；A可定位为支持统计条件下、可调强度的共享嵌入。二者均为**未实现、未证明收益的提案**，普通低秩乘积、支持条件化和Gram核本身不认领首创。

## 实际阅读范围

已读AGENTS、Brain索引及相关偏好/工作流段落、卷积DESIGN_AUDIT与既有RELATED_WORK，并只读检查本目录THREE_OPERATOR_DESIGN_ZH.md的P/A接入。沿用卷积五篇结论，不重复精读。采用literature-review技能的主源核验与方法比较流程。

本侧web.run搜索和open均未返回可读内容；随后实际下载三篇公开PDF并核对标题、作者、方法全文提取文本，不以用户提供链接视为已读证据。页码为PDF文件页序：LRBP第2–6页§2–3、§4和§5.1；CAN第3–6页§3–4；FEAT第3–5页§4.1–4.4，并核对作者`feat.py`的`FEAT._forward`与注意力实现。未运行作者代码，未逐页视觉审图，未独立审读补充材料。

## 三篇核心方法

**[1] Kong、Fowlkes，Low-rank Bilinear Pooling（2016预印本v2）。** 同位置通道外积在空间求和，式(4)、(8)把分类器写成正负低秩项之差：

\[
B(X)=XX^T=\sum_p x_px_p^T,\quad W=U_+U_+^T-U_-U_-^T,
\]
\[
\langle W,B(X)\rangle=\|U_+^TX\|_F^2-\|U_-^TX\|_F^2.
\]

§3再以共享投影分解各类参数：
\[
\min_{P,\{V_k\}}\sum_k\|U_k-PV_k\|_F^2.
\]

这里是未中心化二阶矩，不宜直接叫中心化协方差；低秩首先约束分类器，不能误写成与P完全相同的Hadamard池化。它没有显式跨位置、跨区域配对或当前支持集合条件。P的可辨差异是保留边际并按区域身份、位移带组织乘积；普通投影乘法已属既有技术。若将所有位置无约束独立配对，共现会退化为边际乘积，故配对约束是否保留实际对应才是关键。

**[2] Hou等，CAN，NeurIPS2019。** §3式(1)–(3)先计算类别原型图P与查询图Q的局部余弦矩阵，再从该矩阵生成融合权重：

\[
R_{ij}=\frac{p_i^Tq_j}{\|p_i\|\|q_j\|},\quad R^p=R^T,
\quad w^p=W_2\operatorname{ReLU}(W_1\operatorname{GAP}(R^p)),
\]
\[
A_i^p=\operatorname{softmax}_i((w^p)^Tr_i^p/\tau),\quad
\bar p_i=(1+A_i^p)p_i
\]

查询侧同理。§4先平均同类支持的特征图，每个“类别—查询”对重新加权两端，推理时GAP后按余弦距离分类。训练另含局部分类及全局分类损失；查询伪标签扩充支持是另列的传导式版本，不能混入普通CAN。CAN已覆盖支持关系驱动的空间注意力；A则用支持统计构造全任务共享条件，各图独立映射，不是CAN式逐对匹配。CAN残差倍率为1+A，也不是整个补充分支的关闭门。其成对分数没有自动保证可直接替代本项目PSD核。

**[3] Ye等，FEAT，CVPR2020。** §4.1式(3)–(4)及§4.2定义支持集合适配：

\[
\psi_i=\phi_i+\sum_j\alpha_{ij}W_V^T\phi_j,\qquad
\alpha_{ij}=\operatorname{softmax}_j
\left(\frac{(W_Q^T\phi_i)^T(W_K^T\phi_j)}{\sqrt d}\right).
\]

标准设置的Q/K/V均来自支持集；§4.4与作者代码确认先取同类支持均值，再`self.slf_attn(proto,proto,proto)`，查询保持原嵌入与适配原型比较。代码含残差/LayerNorm，训练另有按类别合并源支持与源查询的辅助正则；不代表测试读取查询标签。FEAT已覆盖支持集合驱动的任务适配。A的区别是无标签均值/方差/覆盖上下文、所有支持及查询采用同一映射、门控服务于附加核。它较FEAT压缩了支持结构，是否足以选择有用证据仍待验证。

## P/A只读判断与待证部分

P的低秩共现与边际并列在数学上成立。主线反例说明同均值可有不同对应，尚不能证明SAR分类收益。197/227的窗口非目标质量24.78%/22.31%只说明可读取；219目标窗口其他候选质量为0，不能将其错判归因于新增3×3直接混入邻船。上述数据来自主线报告，本侧未重新测量或查看原chip。

A在固定支持S下的PSD推导成立，即使门随图像变化：
\[
z_i=\bigoplus_k\sqrt{g_k(x_i;c(S))}\,v_k(x_i;S),\quad
K_{ij}=z_i^Tz_j,\quad a^TKa=\|\sum_i a_iz_i\|^2\ge0.
\]

**未发现使P/A失效的明确逻辑错误；补清两个实现条件：**旧核PSD时，`K_old + lambda K_new`的保证要求`lambda >= 0`；有限logit的sigmoid严格大于0，只能趋近关闭。若声称有效证据也能精确弃用，应明确取零路径，不能把近零称精确恢复。无效证据置零与外部全关消融已有合法零点。

门在中心化/归一化之后，随后不再中心化或单位化，是正确接入。所有附加门为零恢复旧系统；单query门零仍可经支持系数重解受影响，主稿已正确披露。以上是本侧数学判断，不是论文实验结论。收益证据优先沿用主稿的匹配普通二阶池化、保边际破对应，以及标量/图像/支持条件门对照；观察关系变化后仍须验证完整分类净收益与原图案例。

## 主源与本地追溯

- [1] https://arxiv.org/pdf/1611.05109 ，本地`literature/LRBP_2016.pdf/.txt`。
- [2] https://papers.neurips.cc/paper/8655-cross-attention-network-for-few-shot-classification.pdf ，本地`literature/CAN_NeurIPS2019.pdf/.txt`。
- [3] https://openaccess.thecvf.com/content_CVPR_2020/papers/Ye_Few-Shot_Learning_via_Embedding_Adaptation_With_Set-to-Set_Functions_CVPR_2020_paper.pdf ，本地`literature/FEAT_CVPR2020.pdf/.txt`。
- FEAT作者代码：https://raw.githubusercontent.com/Sha-Lab/FEAT/master/model/models/feat.py ，本地`literature/FEAT_author_feat.py.txt`；访问日公开分支副本，未固定提交号。

仅本报告交主线集成归档；literature保持本地。本侧未训练、未安装、未修改主设计或模型，未操作GitHub。
