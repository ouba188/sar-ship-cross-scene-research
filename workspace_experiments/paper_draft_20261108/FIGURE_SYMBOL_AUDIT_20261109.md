# 图3–7模块命名与符号对齐记录

日期：2026-11-09。依据本地AGENTS、FORMULA_FIGURE_ALIGNMENT_ZH.md及当日更新后的MANUSCRIPT_ZH.md公式。仅修改论文流程图副本及其生成脚本、导出物，未修改正文、排版脚本、公式对齐说明或原始PPT。

## 统一命名

| 原标签 | 论文缩写 | 英文名称 | 中文名称 |
|---|---|---|---|
| M1 | CRFB | Class-Response Feedback Block | 类别响应反馈模块 |
| M2 | RGRP | Response-Guided Regional Pooling | 响应引导区域池化 |
| M3 | RCEW | Reference-Class Evidence Writeback | 参考类别证据回写 |

图3紧凑位置使用缩写；内部RGRP用共同标签配合Pool、Write区分池化与回写。图4、图5、图6标题分别给出完整英文名称，图7标明RCEW的通道与空间比较分支。

## 具体修正

| 图 | 修正 |
|---|---|
| 图3 | 更新模块及浅层、深层读出标签；训练主路径的输入、教师/学生分数、选择标记和下界使用样本索引n；损失汇总对齐L_adapt、L_CE,w、0.1 L_KL,correct,w、lambda L_guide,w，标明教师原本正确条件。保留原脚本已有的区域相加、全局拼接修订。 |
| 图4 | 交叉统计T改为C_k，注明每类32维；输入、值、值均值、输出分别使用X_1、V_1、Vbar_1、X'_1；候选响应使用A_k，标准化响应使用s_k，类别总数用K；关系反馈R_1和门G_CRFB与正文一致。均值及关系映射标记L_0、L_k、W_0、W_1，并在残差反馈路径标明0.3。 |
| 图5 | 两处区域beta_r改为pi_r；右端类别响应使用A(u)，左端内容值仍为V_2(u)；2tanh之后的评分改为q_k(u)，避免与此前标准化量s_k混同；补充量使用delta^(2)_(p,r)，标明512维；保留160到512投影、tanh和0.1 stopgrad RMS。 |
| 图6 | 保留行/列边际纠正，采用rho_si与kappa_sj；外积操作直接标明除以M_s；保留T_sij、T^agg_sij、beta_si、vbar_s等参考路径符号；交叉统计C_k不再与此处传输T混用。比较框明确S_sij、B_ic、b_ic、L_sij及c=y_s，保留类外均值而非最大值。输出标为E_c(u)，补出空间分配和0.1 W_out回写关系。 |
| 图7 | 类内/类外均值使用mu_c、mu_(not c)，通道度量使用w_c、空间相似度使用S_sij；保留1+0.5 tanh的有界通道度量。缩排Weighted Cosine Similarity，避开相似度网格。 |

区域/极化索引的省略与正文约定一致。矩阵的区域身份继续由原有分区颜色表达。图3干预候选的局部索引a及原有筛选公式保留，不改变筛选规则。

正文为双栏排版增设的U_0、t_cat、sigma、a_c、t2/t3等中间量没有被强行加入图；图中保留其原有等价运算。

## 可编辑性与导出

- 从只读输入`sources_20261109/author_flowcharts.pptx`重建论文副本，原文件不覆盖。
- 修改位置原有的嵌入式公式对象替换为原生PPT文本框，使用可编辑的上下标字符格式；其余原有形状、图像和公式保留。
- 沿用Microsoft PowerPoint COM导出，不使用其他软件重排PPT。
- 沿用`crop_figure_pages.py`裁去空白，保持PDF矢量内容，同时输出PNG预览。该裁切脚本未修改。
- `symbol_layout_audit.json`记录本轮原生公式文字的实际边界；旋转标签按旋转后的尺寸比较。另逐张查看PowerPoint导出的图形，检查文字遮挡、连接线与边框。
- 未训练、未联网、未执行Git操作。

## 文件

- 修改：`sources_20261109/prepare_paper_figures.ps1`。
- 更新：`author_flowcharts_paper_revision.pptx`。
- 更新：`figures/author_20261109/paper_flowcharts.pdf`、`paper_slide_01.png`至`paper_slide_05.png`。
- 更新：`figures/author_20261109/fig3_author.pdf/png`至`fig7_author.pdf/png`、`figure_manifest.json`。
- 新增：`figures/author_20261109/symbol_layout_audit.json`及本记录。
- 保留：原始`author_flowcharts.pptx`、原始导出`author_flowcharts.pdf`和`slide_01.png`至`slide_05.png`。

图3/6供双栏使用，图5/7供单栏使用；保留既有图文件名，主代理可直接重新编译正文引用。

## 最终状态

2026-11-09 19:25（Asia/Shanghai）导出完成，可用于最终编译。五张图已查看渲染；108个新增原生公式/文字框的边界检查为0处溢出，五页可编辑文本中未检出旧模块标签M1/M2/M3。原始PPT仍为877013字节，修改时间保持2026-11-09 01:00:05。该检查针对本轮文字及导出，不替代主代理对最终单栏/双栏论文页面的排版复核。

后续按用户要求仅补改图4三处并重新导出：响应路径明确为Class contrast及d_k=A_k-mean_(j≠k)A_j，再进入加权标准化；关系汇总标为Mean_k(⊙)，表示逐元素积后按类平均；G_CRFB标签移至门路径上方，以G_CRFB(u)=o(u)tanh(...)说明包含o但不包含X_1，乘法圆点输出不再标G，0.3保留。已查看此次图4渲染，无文字遮挡；当前112个原生文字框检查为0处溢出。PPT、PDF及PNG均已更新，可直接编译。
