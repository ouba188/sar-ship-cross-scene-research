# 核心方法阅读与引用记录

日期：2026-10-23。仅本目录报告；无训练、Git、父文件修改。本轮使用`brainstorming-research-ideas`的问题优先与失效边界视角组织贡献，不执行网络构思扩张。以下为方法阅读而非摘要推断。

## 访问方式与范围

已调用web搜索CSRA/GSoP，并打开arXiv页面/PDF；工具返回空内容，不能据此声称已检索到结果。随后按任务授权，通过requests获取官方arXiv/CVF文档，pypdf在内存解析，未保存完整论文或新增依赖。首次控制台GBK不能打印连字，改用Python UTF-8后成功；不把失败输出计作已读。CSRA与GSoP的核心方法文字/公式已完整读到，未宣称精读两篇全部实验或补充材料。PDF部分图内字体解析乱码，未据乱码推断结构；相关结论由连续方法正文、公式、图注及实现表交叉支撑。

## R1：CSRA

Ke Zhu, Jianxin Wu. **Residual Attention: A Simple but Effective Method for Multi-Label Recognition. ICCV 2021, pp.184–193.** CSRA是模块名称，不是论文正式题名。会议由CVF正式页面核实，不仅依据arXiv年份。

- 官方出版页面：https://openaccess.thecvf.com/content/ICCV2021/html/Zhu_Residual_Attention_A_Simple_but_Effective_Method_for_Multi-Label_Recognition_ICCV_2021_paper.html
- 本轮实际方法来源：https://arxiv.org/pdf/2108.02456
- 实际取到10页PDF，页首注明arXiv:2108.02456v2，19 Aug 2021。读PDF第1–5页；重点第3–5页完整第3.1–3.4节，以及第4.1节训练设置。未读完第6–10页或supplement。
- 第3.2节式(2)–(6)：用分类权重m_c投影各位置特征x_u，以softmax(T m_c^T x_u)给空间权重；加权特征a_c与普通GAP特征g按g+lambda*a_c相加，再由同一m_c分类。
- 第3.3节式(7)–(12)：等价为GAP分数加空间softmax加权分数；T乘在logit上，T趋于无穷得到max分支，不是通常写法中除以温度。对归一化组合的解释不等于物理定位证明。
- 第3.4节及图3：多个head有各自1x1分类器、不同固定T序列，共享lambda，分支logit相加；不是所有head共用唯一分类W。文中另用分类权重单位范数归一化改善收敛。
- 第4.1节：训练版端到端BCE/SGD并微调backbone。第1节另有无需再训练的mean+max测试时插件。故不能把“需训练”作为与我们的唯一差异。

## R2：GSoP-Net

Zilin Gao, Jiangtao Xie, Qilong Wang, Peihua Li. **Global Second-Order Pooling Convolutional Networks. CVPR 2019, pp.3024–3033.** 会议信息由CVF页面核实。

- 官方页面：https://openaccess.thecvf.com/content_CVPR_2019/html/Gao_Global_Second-Order_Pooling_Convolutional_Networks_CVPR_2019_paper.html
- 本轮实际方法来源：https://openaccess.thecvf.com/content_CVPR_2019/papers/Gao_Global_Second-Order_Pooling_Convolutional_Networks_CVPR_2019_paper.pdf
- 10页PDF，已读PDF第1、3、4、5页，即印刷3024、3026–3028页。第3.1–3.4节完整读到，第5页表2/3及第4节训练设置也读到；未将第2、6–10页或所有消融称为已读。
- 第3.1节：1x1降维后计算全图通道协方差，逐行归一化、逐行卷积与非线性映射产生通道权重，缩放原特征。表2给出conv4_x实例，降到128通道后最终输出回原1024通道尺度。
- 第3.2节：位置版把空间压为固定网格，计算位置间协方差、映射成位置权重，再上采样缩放特征；本文的两种版本不要混为一个算子。
- 第3.4节：默认通道降到128，位置网格8x8；逐行BN/group conv；有输入到缩放输出的shortcut。GSoP-Net1末端GSoP+GAP，GSoP-Net2采用矩阵幂归一化协方差表示。
- 第4节：自然图像ImageNet/CIFAR训练，图像缩放裁剪，分类监督训练网络；不是固定SAR源OOF类别方向上的末端统计读出。本轮选GSoP而非再扩读Bilinear CNN，因其直接覆盖“全局二阶统计如何改变特征利用”的最邻近问题。

## R3/R4：复用的SAR primary阅读笔记

来源为本地`../literature_comparison/FINDINGS_ZH.md`，已完整读取。以下原文阅读范围是原笔记作者此前完成的范围，**本轮未重新下载或独立逐页复读这两篇**。不把他人的实际阅读记为本轮新阅读。

R3：He等，**CPINet: Towards A Novel Cross-Polarimetric Interaction Network for Dual-Polarized SAR Ship Classification**，Remote Sensing 2024, 16, 3479。DOI：https://doi.org/10.3390/rs16183479 。原笔记记录读第1–13页，重点9–13，全文24页未全部读。

记录的核心方法：MO-SE由一路的GAP和分组双线性统计产生权重、调制另一路；FBC在对应位置做低秩双线性交互、带符号软阈值与最大池化；多尺度/单极化分支监督，GradNorm协调四项CE。它已直接覆盖双极化、一二阶统计及局部带符号非线性的相关思想。

R4：Garaiman与Radoi，**A Lightweight and Real-Time Dual-Polarization Fusion Framework for SAR Ship Classification**，Remote Sensing 2026, 18, 1129。DOI：https://doi.org/10.3390/rs18081129 。原笔记记录读第1–13、17–23页，未读14–16页。

记录的核心方法：两个MobileViTv2分支GAP各512维，拼为1024维分类；表4对比求和、平均、1x1卷积和拼接。256缩放、80/20划分；原笔记所读设置未明确采集/MMSI隔离，不等于证明其未隔离或存在泄漏。其普通GAP融合是我们必须认真对待的直接近邻。

## 本地实证来源

- `../MODULE_METHOD_ZH.md`：真实209参数local方法，而非历史226参数carrier头。
- `../strong_calibration_increment/FINDINGS_ZH.md`及`pooled.csv`：18头强B增量、类别代价、源验证及原图记录。
- `../mean_only_control/FINDINGS_ZH.md`及`paired_metrics.csv`：9头均值控制和源/test互补计数。新结果优先于MODULE_METHOD中尚未更新的“均值对照待做”。
- 最新增补：`../joint_distribution_readout/PROTOCOL_ZH.md`、`pooled.csv`、`summary.json`已只读，18头完成；joint273与additive266同信息近容量。只据已落盘指标更新novelty边界，不冒称完成其原图/逐类/CI分析，不重训或新增控制。
- `../free_information_control/FINDINGS_ZH.md`：早期较弱B下aggregate/constant控制；不能冒充强B上的新控制。
- `../score_matched_moment_diagnosis/centered_correction/FINDINGS_ZH.md`：oldAD固定free_real的centered匹配诊断；不能冒充强B上mean/local的条件比较。

本次没有重做数值实验、重绘图或重看全部图像；图像可见事实与案例变化明确复用已完成的原图报告。正式引用可按上述作者/题名/会议信息录入；本目录未复制整篇论文文本。
