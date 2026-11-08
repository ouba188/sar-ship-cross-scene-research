# 新增参考文献与方法对应关系

核查日期：2026-11-09。新增 **13篇** 原始论文，均在原稿refs1-14之外。本文档和同目录JSON为独立支线交付；未修改主稿、模型或实验，未推送Git。A15-A27仅为整合标识，不代表已在主稿中编号。

## 整合结论

| 主稿部分 | 新增文献 | 核心对应关系 |
|---|---|---|
| M1类别响应、均值与高阶统计 | A15 CAM；A16 SENet；A17 GSoP；A18 SRM | 类别响应、均值通道门、协方差反馈及均值/标准差重标定均已有基础。主稿应比较相对类响应与局部值交叉统计的定义、反馈位置及其与M2的接口。 |
| M2区域池化、上下文回写与几何读出 | A19 A2-Nets；A20 GCNet；A21 R-FCN | 区分汇总后自适应再分配、全局上下文广播、位置敏感类别读出；主稿是OBB相对区域汇总、区域限定回写及分开的摘要读出。 |
| M3局部参考、原型与类条件匹配 | A22 ProtoNet；A23 DN4；A24 CAN；A25 FRN | 对照全局原型、局部到类近邻、类查询交叉注意和类条件重构；主稿重点是保留参考选择的边际聚合及类别证据回写。 |
| 受控干预与解释训练 | A26 RSC；A27输入梯度正则 | 区分训练期屏蔽高梯度特征、惩罚输入敏感性，以及主稿依据受控操作筛选并保护教师类别优势。 |

**M3按当前稿件固定口径比较：** \(\widetilde T_{ab}=\rho_a\kappa_b/M\)取消精细值传递；仍保留原传输、参考选择和类内权重，不能称为“没有匹配”或“普通均值原型”。参考聚合后仍可因查询相关权重及查询值而形成局部类别证据。当前贡献表述不强调精细对应增益，也不将缓存的固定锚点残差写成与新版M1实时联合学习。

这些文献建立方法基础与可核查差异，**不提供本文模块优于前人的实验证据，也不构成全领域首次性的证明**。通用视觉原文仅用于算子比较，不能以其数据集结果替代SAR同协议对照。已有CPINet、SCRC-Net、DeepEMD及refs1-14不重做。

## 可直接整合的新增表

每行同时提供完整bibliography、primary URL、可入稿方法比较与放置位置。书目中的全部作者姓名另在JSON中完整展开。

| ID / 方法 | 完整bibliography | Primary URL | 方法差异与可入稿比较 | 放置节 |
|---|---|---|---|---|
| A15 / CAM | B. Zhou, A. Khosla, A. Lapedriza, A. Oliva, and A. Torralba, "Learning Deep Features for Discriminative Localization," in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 2921-2929, doi: 10.1109/CVPR.2016.319. | [原始来源](https://openaccess.thecvf.com/content_cvpr_2016/html/Zhou_Learning_Deep_Features_CVPR_2016_paper.html) | CAM通过将分类权重投射到最后卷积特征获得类别响应图。本文在中间层生成受辅助分类监督约束的候选类响应，并将其用于特征反馈及区域内取证，因此比较重点是响应参与后续计算的方式。 | 2.2；4.2；4.3.2 |
| A16 / SENet | J. Hu, L. Shen, and G. Sun, "Squeeze-and-Excitation Networks," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2018, pp. 7132-7141, doi: 10.1109/CVPR.2018.00745. | [原始来源](https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.html) | SENet利用全局均值摘要生成空间共享的通道权重。本文保留均值上下文，并结合相对类别响应与局部值的共同变化生成位置相关反馈，研究这一接口对区域读出的作用。 | 2.2；4.2 |
| A17 / GSoP-Net | Z. Gao, J. Xie, Q. Wang, and P. Li, "Global Second-Order Pooling Convolutional Networks," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 3024-3033, doi: 10.1109/CVPR.2019.00314. | [原始来源](https://openaccess.thecvf.com/content_CVPR_2019/html/Gao_Global_Second-Order_Pooling_Convolutional_Networks_CVPR_2019_paper.html) | GSoP-Net已将全局二阶统计引入中间层，并据此进行通道或位置重标定。本文使用局部值与相对候选类别响应之间的交叉统计构造反馈，而非直接计算完整通道或位置协方差矩阵。 | 2.2；4.2 |
| A18 / SRM | H. Lee, H.-E. Kim, and H. Nam, "SRM: A Style-Based Recalibration Module for Convolutional Neural Networks," in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2019, pp. 1854-1862, doi: 10.1109/ICCV.2019.00194. | [原始来源](https://openaccess.thecvf.com/content_ICCV_2019/html/Lee_SRM_A_Style-Based_Recalibration_Module_for_Convolutional_Neural_Networks_ICCV_2019_paper.html) | SRM将通道均值与标准差共同用于特征重标定，说明均值以外的统计量已有注意机制基础。本文进一步显式保留候选类别维度，并利用响应与局部值的交叉统计形成空间反馈。 | 2.2；4.2 |
| A19 / A2-Nets | Y. Chen, Y. Kalantidis, J. Li, S. Yan, and J. Feng, "A^2-Nets: Double Attention Networks," in Advances in Neural Information Processing Systems, vol. 31, 2018, pp. 352-361. | [原始来源](https://ai.meta.com/research/publications/a2-nets-double-attention-networks/) | A2-Nets先将全空间信息汇总为多个描述，再通过位置相关注意将其分配回特征图。本文M2以目标几何区域限定汇总与回写范围，并将内部回写与供分类使用的区域摘要读出分开。 | 2.2；4.3.1 |
| A20 / GCNet | Y. Cao, J. Xu, S. Lin, F. Wei, and H. Hu, "GCNet: Non-Local Networks Meet Squeeze-Excitation Networks and Beyond," in Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops (ICCVW), 2019, pp. 1971-1980, doi: 10.1109/ICCVW.2019.00246. | [原始来源](https://openaccess.thecvf.com/content_ICCVW_2019/html/NeurArch/Cao_GCNet_Non-Local_Networks_Meet_Squeeze-Excitation_Networks_and_Beyond_ICCVW_2019_paper.html) | GCNet通过注意池化、瓶颈变换和加法广播将全局上下文注入中间特征。本文M2按框内、轴端和侧向分别汇总，并仅向相应区域回写，以保留目标相对的证据来源。 | 2.2；4.3.1 |
| A21 / R-FCN | J. Dai, Y. Li, K. He, and J. Sun, "R-FCN: Object Detection via Region-based Fully Convolutional Networks," in Advances in Neural Information Processing Systems, vol. 29, 2016, pp. 379-387. | [原始来源](https://proceedings.neurips.cc/paper/2016/hash/577ef1154f3240ad5b9b413aa7346a1e-Abstract.html) | R-FCN以位置敏感得分图和对应RoI网格池化联合编码类别与相对位置。本文面向已定位目标，采用OBB相对的框内、轴端和侧向区域组织特征读出，并将区域摘要与内部上下文回写分别设置。 | 2.2；4.3.1 |
| A22 / Prototypical Networks | J. Snell, K. Swersky, and R. Zemel, "Prototypical Networks for Few-shot Learning," in Advances in Neural Information Processing Systems, vol. 30, 2017, pp. 4077-4087. | [原始来源](https://proceedings.neurips.cc/paper/2017/hash/cb8da6767461f2812ae4290eac7cbc42-Abstract.html) | Prototypical Networks通过类内支持嵌入均值及查询到原型的距离完成分类。本文M3在局部区域内形成查询相关的类参考表示，并将比较得到的类别证据回写中间特征，而非直接以全局原型距离作为最终输出。 | 2.3；4.4.2 |
| A23 / DN4 | W. Li, L. Wang, J. Xu, J. Huo, Y. Gao, and J. Luo, "Revisiting Local Descriptor Based Image-To-Class Measure for Few-Shot Learning," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 7260-7268, doi: 10.1109/CVPR.2019.00743. | [原始来源](https://openaccess.thecvf.com/content_CVPR_2019/html/Li_Revisiting_Local_Descriptor_Based_Image-To-Class_Measure_for_Few-Shot_Learning_CVPR_2019_paper.html) | DN4将同类支持图的局部描述组成类内池，累加查询局部描述的近邻相似度完成分类。本文M3保留区域与参考权重，并把类条件比较转为可继续加工的类别证据场，其差异落在消息内容与回写位置。 | 2.3；4.4.2 |
| A24 / CAN | R. Hou, H. Chang, B. Ma, S. Shan, and X. Chen, "Cross Attention Network for Few-shot Classification," in Advances in Neural Information Processing Systems, vol. 32, 2019, pp. 4003-4014. | [原始来源](https://proceedings.neurips.cc/paper/2019/hash/01894d6f048493d2cacde3c579c315a3-Abstract.html) | CAN利用类参考与查询的局部相关性生成交叉注意，并在归纳推理中比较注意加权后的表示。本文M3将参考比较进一步转为类别证据通道，在layer3之后回写供后续骨干处理，重点检验这种消息形式及其接入作用。 | 2.3；4.4.3 |
| A25 / FRN | D. Wertheimer, L. Tang, and B. Hariharan, "Few-Shot Classification With Feature Map Reconstruction Networks," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021, pp. 8012-8021, doi: 10.1109/CVPR46437.2021.00792. | [原始来源](https://openaccess.thecvf.com/content/CVPR2021/html/Wertheimer_Few-Shot_Classification_With_Feature_Map_Reconstruction_Networks_CVPR_2021_paper.html) | FRN通过类内支持局部特征对查询特征图进行岭回归重构，并以重构误差形成类别分数。本文M3采用保留传输边际的参考值聚合，将局部类距离转为空间类别证据回写，因此不以精细值对应或整图重构增益作为当前主张。 | 2.3；4.4.2；4.4.3 |
| A26 / RSC | Z. Huang, H. Wang, E. P. Xing, and D. Huang, "Self-challenging Improves Cross-Domain Generalization," in Computer Vision - ECCV 2020, Lecture Notes in Computer Science, vol. 12347, 2020, pp. 124-140, doi: 10.1007/978-3-030-58536-5_8. | [原始来源](https://link.springer.com/chapter/10.1007/978-3-030-58536-5_8) | RSC在训练中抑制对真实类预测梯度较高的特征，使模型利用其余表示。本文依据限定输入干预及均值对照筛选类别优势保护对象，关注何种操作确有判别收益，而不直接把主导响应统一视为应删除的信息。 | 2.4；4.6 |
| A27 / Input-gradient Regularization | A. S. Ross and F. Doshi-Velez, "Improving the Adversarial Robustness and Interpretability of Deep Neural Networks by Regularizing Their Input Gradients," in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 32, no. 1, 2018, pp. 1660-1669, doi: 10.1609/aaai.v32i1.11504. | [原始来源](https://ojs.aaai.org/index.php/AAAI/article/view/11504) | Ross和Doshi-Velez通过惩罚分类损失的输入梯度，将局部敏感性约束纳入训练。本文按具体极化、区域与尺度操作的判别效应筛选优势保护对象，约束的是教师类别分差下界，二者作用于不同的训练量。 | 2.4；4.6 |

## 逐篇核查记录

以下“阅读范围”是实际读到的原文页/节，不代表精读全部实验和附录。全文访问受限时明确标注作者版本；不以搜索摘要替代方法核查。

### A15 CAM

- **完整书目：** B. Zhou, A. Khosla, A. Lapedriza, A. Oliva, and A. Torralba, "Learning Deep Features for Discriminative Localization," in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 2921-2929, doi: 10.1109/CVPR.2016.319.
- **Primary URL：** <https://openaccess.thecvf.com/content_cvpr_2016/html/Zhou_Learning_Deep_Features_CVPR_2016_paper.html>
- **正式全文：** <https://openaccess.thecvf.com/content_cvpr_2016/papers/Zhou_Learning_Deep_Features_CVPR_2016_paper.pdf>
- **书目核实：** <https://api.crossref.org/works/10.1109/CVPR.2016.319>
- **实际阅读范围：** CVF正式PDF第2-3页，Sec. 2、Fig. 2、Eqs. (1)-(2)；官网作者信息和Crossref书目。
- **作用位置：** 最后卷积层经过全局平均池化和线性分类器；将分类器权重投回最后卷积图，形成每个类别的空间响应。
- **核对到的方法：** CAM由各通道特征图的类别权重和构成，其空间和对应分类分数。图像级分类监督产生可定位的类别响应；该原文的CAM并不是插在中间卷积中的特征反馈门。
- **与当前主稿的差异：** M1在layer2末块内部生成全部候选类响应，中心化、标准化后与局部值形成交叉统计，反馈特征并指导M2取证；其响应并非最终分类头分数的直接空间分解。
- **可入稿比较（1-2句）：** CAM通过将分类权重投射到最后卷积特征获得类别响应图。本文在中间层生成受辅助分类监督约束的候选类响应，并将其用于特征反馈及区域内取证，因此比较重点是响应参与后续计算的方式。
- **建议放置：** 2.2 双极化与分层证据融合，新增类别响应段；4.2 M1类别响应反馈；4.3.2 由M1响应指导浅层取证。
- **写作边界：** 不能把类别响应图或利用图像级标签形成空间响应本身写成首次提出。

### A16 SENet

- **完整书目：** J. Hu, L. Shen, and G. Sun, "Squeeze-and-Excitation Networks," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2018, pp. 7132-7141, doi: 10.1109/CVPR.2018.00745.
- **Primary URL：** <https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.html>
- **正式全文：** <https://openaccess.thecvf.com/content_cvpr_2018/papers/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.pdf>
- **书目核实：** <https://api.crossref.org/works/10.1109/CVPR.2018.00745>
- **实际阅读范围：** CVF正式PDF第3-4页，Secs. 3.1-3.3、Eqs. (2)-(4)、Fig. 3；CVPR版官网和DOI元数据。
- **作用位置：** 以全局平均池化提取通道摘要，经两层瓶颈和sigmoid得到通道权重；SE-ResNet在残差分支与identity相加前缩放该分支。
- **核对到的方法：** 每个通道得到一个空间共享的缩放系数；标准SE并不形成有标签的候选类别响应，也不在区域内分别归一化注意。原文允许接入不同骨干。
- **与当前主稿的差异：** M1保留均值支路，同时增加值特征与相对类别响应的交叉统计和随位置变化的反馈；当前插入点为layer2[3]的conv2激活后、conv3前，并非SE默认残差输出位置。
- **可入稿比较（1-2句）：** SENet利用全局均值摘要生成空间共享的通道权重。本文保留均值上下文，并结合相对类别响应与局部值的共同变化生成位置相关反馈，研究这一接口对区域读出的作用。
- **建议放置：** 2.2 统计注意基础；4.2 M1类别响应反馈。
- **写作边界：** 采用CVPR 2018三作者版本；不混入TPAMI扩展版作者列表，也不将扩展版位置消融归入CVPR原版。

### A17 GSoP-Net

- **完整书目：** Z. Gao, J. Xie, Q. Wang, and P. Li, "Global Second-Order Pooling Convolutional Networks," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 3024-3033, doi: 10.1109/CVPR.2019.00314.
- **Primary URL：** <https://openaccess.thecvf.com/content_CVPR_2019/html/Gao_Global_Second-Order_Pooling_Convolutional_Networks_CVPR_2019_paper.html>
- **正式全文：** <https://openaccess.thecvf.com/content_CVPR_2019/papers/Gao_Global_Second-Order_Pooling_Convolutional_Networks_CVPR_2019_paper.pdf>
- **书目核实：** <https://api.crossref.org/works/10.1109/CVPR.2019.00314>
- **实际阅读范围：** CVF正式PDF第3-4页，Secs. 3.1-3.4、Fig. 1；首末页确认CVF页码3024-3033。
- **作用位置：** 可在网络中间层插入GSoP块：先降维，再计算通道协方差并经行归一化和卷积映射产生缩放；另给出位置协方差版本。
- **核对到的方法：** GSoP不只是在网络末端输出二阶描述，还利用中间层协方差调节特征。位置版先缩小空间尺寸，产生位置相关缩放；不能笼统说已有二阶注意均无空间作用。
- **与当前主稿的差异：** M1的c_k是低维值与标准化相对类响应之间的交叉统计，既非完整通道协方差，也非位置两两协方差；它保留独立均值路径并用于类别相关反馈。
- **可入稿比较（1-2句）：** GSoP-Net已将全局二阶统计引入中间层，并据此进行通道或位置重标定。本文使用局部值与相对候选类别响应之间的交叉统计构造反馈，而非直接计算完整通道或位置协方差矩阵。
- **建议放置：** 2.2 高阶统计注意段，紧接CPINet；4.2 式(4)之后。
- **写作边界：** 不能宣称首次中间层二阶注意；主稿现有证据也不支持将c_k独有准确率收益写成已证实。
- **书目差异说明：** Crossref登记页码为3019-3028；本交付采用CVF正式PDF印刷页码3024-3033。DOI和题名、作者一致。

### A18 SRM

- **完整书目：** H. Lee, H.-E. Kim, and H. Nam, "SRM: A Style-Based Recalibration Module for Convolutional Neural Networks," in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2019, pp. 1854-1862, doi: 10.1109/ICCV.2019.00194.
- **Primary URL：** <https://openaccess.thecvf.com/content_ICCV_2019/html/Lee_SRM_A_Style-Based_Recalibration_Module_for_Convolutional_Neural_Networks_ICCV_2019_paper.html>
- **正式全文：** <https://openaccess.thecvf.com/content_ICCV_2019/papers/Lee_SRM_A_Style-Based_Recalibration_Module_for_Convolutional_Neural_Networks_ICCV_2019_paper.pdf>
- **书目核实：** <https://api.crossref.org/works/10.1109/ICCV.2019.00194>
- **实际阅读范围：** CVF正式PDF第3-4页，Secs. 3.1-3.3、Eqs. (1)-(9)、Fig. 2；CVF与Crossref书目。
- **作用位置：** 在中间特征上逐通道提取空间均值和标准差，再以通道独立映射、BN、sigmoid产生空间共享的通道缩放。
- **核对到的方法：** SRM使用两个逐通道统计量，主要方案没有计算完整跨通道协方差；其style名称指所用统计表示，不等同于已识别某种物理风格因子。
- **与当前主稿的差异：** M1的新增关系项具有候选类别轴，度量值与类别响应的共同变化，并在空间上反馈；不是将每通道均值和标准差拼接后做普通重标定。
- **可入稿比较（1-2句）：** SRM将通道均值与标准差共同用于特征重标定，说明均值以外的统计量已有注意机制基础。本文进一步显式保留候选类别维度，并利用响应与局部值的交叉统计形成空间反馈。
- **建议放置：** 2.2 均值与高阶统计注意段；4.2 统计量定义附近。
- **写作边界：** 不能把增加第二个统计量直接解释成新机制，或将SAR响应统计自动命名为物理散射因素。

### A19 A2-Nets

- **完整书目：** Y. Chen, Y. Kalantidis, J. Li, S. Yan, and J. Feng, "A^2-Nets: Double Attention Networks," in Advances in Neural Information Processing Systems, vol. 31, 2018, pp. 352-361.
- **Primary URL：** <https://ai.meta.com/research/publications/a2-nets-double-attention-networks/>
- **正式全文：** <https://papers.neurips.cc/paper/7318-a2-nets-double-attention-networks.pdf>
- **实际补充精读全文：** <https://arxiv.org/pdf/1810.11579>
- **书目核实：** <https://proceedings.neurips.cc/paper_files/paper/2018/file/e165421110ba03099a1c0393373c5b43-Metadata.json>
- **实际阅读范围：** NeurIPS原文已读取前部与官方metadata；作者arXiv原文第3-5页，Secs. 2.1-2.4、Eqs. (2)-(7)、Figs. 1-2，完整核对汇总和再分配。
- **作用位置：** 第一步对全空间特征进行注意池化形成紧凑全局描述集合，第二步以各位置的另一组权重选取并再分配，升维后残差加回特征图。
- **核对到的方法：** 主要计算为[phi(X) softmax(theta(X))^T] softmax(rho(X))；二阶注意池化来自双线性视角，不是M1的中心化类别交叉统计。
- **与当前主稿的差异：** M2内部按OBB相对区域池化，并经区域专属变换回写相应区域；浅层类别引导读出输出区域摘要。两者不能合称一个A2式全空间汇总再分配块。
- **可入稿比较（1-2句）：** A2-Nets先将全空间信息汇总为多个描述，再通过位置相关注意将其分配回特征图。本文M2以目标几何区域限定汇总与回写范围，并将内部回写与供分类使用的区域摘要读出分开。
- **建议放置：** 2.2 区域上下文与回写段；4.3.1 区域身份与内部回写。
- **写作边界：** 汇总后再回写、两阶段注意及残差接入均有先例；不以A2的名称证明M1统计是同一种二阶量。
- **书目差异说明：** 采用大会官方Metadata.json的352-361；部分引用记录写350-359，未混用。大会记录未提供出版DOI，故doi为null；arXiv编号不是大会DOI。

### A20 GCNet

- **完整书目：** Y. Cao, J. Xu, S. Lin, F. Wei, and H. Hu, "GCNet: Non-Local Networks Meet Squeeze-Excitation Networks and Beyond," in Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops (ICCVW), 2019, pp. 1971-1980, doi: 10.1109/ICCVW.2019.00246.
- **Primary URL：** <https://openaccess.thecvf.com/content_ICCVW_2019/html/NeurArch/Cao_GCNet_Non-Local_Networks_Meet_Squeeze-Excitation_Networks_and_Beyond_ICCVW_2019_paper.html>
- **正式全文：** <https://openaccess.thecvf.com/content_ICCVW_2019/papers/NeurArch/Cao_GCNet_Non-Local_Networks_Meet_Squeeze-Excitation_Networks_and_Beyond_ICCVW_2019_paper.pdf>
- **书目核实：** <https://api.crossref.org/works/10.1109/ICCVW.2019.00246>
- **实际阅读范围：** CVF正式PDF第3-5页，Secs. 3.1-3.2与4.1-4.3，Fig. 4及Eqs. (3)-(5)；本地既有位置消融精读仅作辅助。
- **作用位置：** 全空间注意加权获得一个上下文向量，经瓶颈、LayerNorm和ReLU变换后，以加法广播至各位置。
- **核对到的方法：** GC块共享全局上下文，标准形式不是按每个查询位置分别产生一份上下文，也不是SE的乘法通道门。其研究包含不同接入位置，但不能据此等同当前M2的内部位置。
- **与当前主稿的差异：** M2的式(7)为三个几何区域各自池化、变换，再由对应区域掩膜限制回写；layer3[5]中的更新发生在conv1激活后、conv2前。
- **可入稿比较（1-2句）：** GCNet通过注意池化、瓶颈变换和加法广播将全局上下文注入中间特征。本文M2按框内、轴端和侧向分别汇总，并仅向相应区域回写，以保留目标相对的证据来源。
- **建议放置：** 2.2 区域上下文与回写段；4.3.1 式(7)附近。
- **写作边界：** 必须标注ICCV Workshops，不能写成ICCV主会；不能把当前M2接入位置称为GCNet位置消融的直接复现。

### A21 R-FCN

- **完整书目：** J. Dai, Y. Li, K. He, and J. Sun, "R-FCN: Object Detection via Region-based Fully Convolutional Networks," in Advances in Neural Information Processing Systems, vol. 29, 2016, pp. 379-387.
- **Primary URL：** <https://proceedings.neurips.cc/paper/2016/hash/577ef1154f3240ad5b9b413aa7346a1e-Abstract.html>
- **正式全文：** <https://proceedings.neurips.cc/paper_files/paper/2016/file/577ef1154f3240ad5b9b413aa7346a1e-Paper.pdf>
- **实际补充精读全文：** <https://arxiv.org/pdf/1605.06409>
- **书目核实：** <https://proceedings.neurips.cc/paper_files/paper/2016/file/577ef1154f3240ad5b9b413aa7346a1e-Metadata.json>
- **实际阅读范围：** 大会官方metadata内全文Sec. 2；作者arXiv原文第2-4页，Figs. 1-2、位置敏感RoI池化式(1)和后续投票。
- **作用位置：** 最后卷积层输出每类k×k个位置敏感得分图；RoI内每个网格只从对应位置通道池化，然后投票形成RoI类别分数。
- **核对到的方法：** 这是检测末端的类别与相对位置联合读出，RoI池化后无可学习卷积或全连接层。它没有本文的OBB框外轴端／侧向区域，也不是区域摘要空间回写。
- **与当前主稿的差异：** M2针对已定位船只，按OBB构造框内及框外相对区域，从特征值形成摘要并另有内部回写；不是固定k×k类别得分图的检测投票。
- **可入稿比较（1-2句）：** R-FCN以位置敏感得分图和对应RoI网格池化联合编码类别与相对位置。本文面向已定位目标，采用OBB相对的框内、轴端和侧向区域组织特征读出，并将区域摘要与内部上下文回写分别设置。
- **建议放置：** 2.2 几何区域读出段，与SCRC-Net并列但区分检测任务；4.3.1 区域定义。
- **写作边界：** 不将检测性能与船型分类性能作直接比较；相对位置保持和类别相关区域读出不是本文首次引入。
- **书目差异说明：** 大会官方metadata核实379-387；未提供出版DOI，doi为null。

### A22 Prototypical Networks

- **完整书目：** J. Snell, K. Swersky, and R. Zemel, "Prototypical Networks for Few-shot Learning," in Advances in Neural Information Processing Systems, vol. 30, 2017, pp. 4077-4087.
- **Primary URL：** <https://proceedings.neurips.cc/paper/2017/hash/cb8da6767461f2812ae4290eac7cbc42-Abstract.html>
- **正式全文：** <https://proceedings.neurips.cc/paper/6996-prototypical-networks-for-few-shot-learning.pdf>
- **书目核实：** <https://proceedings.neurips.cc/paper_files/paper/2017/file/cb8da6767461f2812ae4290eac7cbc42-Metadata.json>
- **实际阅读范围：** 大会PDF第2-3页，Secs. 2.1-2.3、Eqs. (1)-(2)、Algorithm 1；官方metadata核实作者和页码。
- **作用位置：** 在嵌入空间对每类支持样本求均值，依据查询到各类原型的距离直接生成分类概率；以episode构造训练任务。
- **核对到的方法：** 类原型聚合和距离证据是已有基础。原文few-shot任务需识别训练阶段未见类别；不等同本文固定已知四类及32个源参考的闭集预测。
- **与当前主稿的差异：** M3在区域和局部槽位层保留类别组织及查询相关参考权重，将候选类距离中心化后投回中间空间场，交给后续骨干处理。
- **可入稿比较（1-2句）：** Prototypical Networks通过类内支持嵌入均值及查询到原型的距离完成分类。本文M3在局部区域内形成查询相关的类参考表示，并将比较得到的类别证据回写中间特征，而非直接以全局原型距离作为最终输出。
- **建议放置：** 2.3 局部参考比较与类别证据，开头基础段；4.4.2 类内参考聚合。
- **写作边界：** 不能把类内聚合、原型或负距离作为独立原创；固定源参考库不应写成少样本新类别测试。
- **书目差异说明：** 大会metadata核实4077-4087，未提供出版DOI；不把数据库的10.5555标识或arXiv DOI冒充已核实大会DOI。

### A23 DN4

- **完整书目：** W. Li, L. Wang, J. Xu, J. Huo, Y. Gao, and J. Luo, "Revisiting Local Descriptor Based Image-To-Class Measure for Few-Shot Learning," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 7260-7268, doi: 10.1109/CVPR.2019.00743.
- **Primary URL：** <https://openaccess.thecvf.com/content_CVPR_2019/html/Li_Revisiting_Local_Descriptor_Based_Image-To-Class_Measure_for_Few-Shot_Learning_CVPR_2019_paper.html>
- **正式全文：** <https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Revisiting_Local_Descriptor_Based_Image-To-Class_Measure_for_Few-Shot_Learning_CVPR_2019_paper.pdf>
- **书目核实：** <https://api.crossref.org/works/10.1109/CVPR.2019.00743>
- **实际阅读范围：** CVF正式PDF第3-4页，Secs. 3.2-3.4、Fig. 1、Eqs. (1)-(2)；官网和PDF首末页核实页码。
- **作用位置：** 从全卷积嵌入保留密集局部描述，将同类支持图的描述合并入池，逐查询描述检索类内k近邻并累加余弦相似度作为类别分数。
- **核对到的方法：** DN4是image-to-class局部度量，不是先把每张图压成一个全局原型；其类分数来自非参数近邻汇总。
- **与当前主稿的差异：** M3保留区域身份、参考选择及类内权重，使用保留传输边际的区域参考值，再将距离关系变成中间类别场；最终分类还经过layer4和融合头。
- **可入稿比较（1-2句）：** DN4将同类支持图的局部描述组成类内池，累加查询局部描述的近邻相似度完成分类。本文M3保留区域与参考权重，并把类条件比较转为可继续加工的类别证据场，其差异落在消息内容与回写位置。
- **建议放置：** 2.3 在全局原型之后、DeepEMD之前；4.4.2 保留参考选择而取消精细值耦合。
- **写作边界：** 不要将局部参考比较仅归于DeepEMD，也不要声称局部到类别匹配是新提出的。
- **书目差异说明：** CVF官网和原PDF为7260-7268；Crossref登记7253-7260。采用CVF版本页码。

### A24 CAN

- **完整书目：** R. Hou, H. Chang, B. Ma, S. Shan, and X. Chen, "Cross Attention Network for Few-shot Classification," in Advances in Neural Information Processing Systems, vol. 32, 2019, pp. 4003-4014.
- **Primary URL：** <https://proceedings.neurips.cc/paper/2019/hash/01894d6f048493d2cacde3c579c315a3-Abstract.html>
- **正式全文：** <https://proceedings.neurips.cc/paper/2019/file/01894d6f048493d2cacde3c579c315a3-Paper.pdf>
- **实际补充精读全文：** <https://arxiv.org/pdf/1910.07677>
- **书目核实：** <https://proceedings.neurips.cc/paper_files/paper/2019/file/01894d6f048493d2cacde3c579c315a3-Metadata.json>
- **实际阅读范围：** 大会PDF前部和官方metadata；作者arXiv原文第3-6页，Sec. 3、Fig. 2、Eqs. (1)-(3)、Sec. 4及归纳/传导推理段。
- **作用位置：** 在最后卷积后，对类特征图和查询图计算局部余弦相关矩阵，由元融合生成双向空间注意，以1+A缩放特征；归纳推理全局平均后做余弦最近类判断。
- **核对到的方法：** 类参考特征与查询不是独立读取；原文已有查询相关、类条件的局部注意。传导推理是另一路径，会以伪标签查询扩充支持；不能据此把全部CAN都称为传导方法。
- **与当前主稿的差异：** M3回写的是中心化类距离证据而非对原外观特征做双向加权，并位于layer3后、layer4前；当前固定源参考不吸收目标查询伪标签。
- **可入稿比较（1-2句）：** CAN利用类参考与查询的局部相关性生成交叉注意，并在归纳推理中比较注意加权后的表示。本文M3将参考比较进一步转为类别证据通道，在layer3之后回写供后续骨干处理，重点检验这种消息形式及其接入作用。
- **建议放置：** 2.3 类条件局部匹配段；4.4.3 类别证据场及空间回写。
- **写作边界：** 不能声称首次类条件局部注意，也不能把当前M3描述成实时与新M1联合学习的参考关系。
- **书目差异说明：** 大会官方metadata为4003-4014；采用该字段，不采用其他索引的4005-4016。大会未提供出版DOI，doi为null。

### A25 FRN

- **完整书目：** D. Wertheimer, L. Tang, and B. Hariharan, "Few-Shot Classification With Feature Map Reconstruction Networks," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021, pp. 8012-8021, doi: 10.1109/CVPR46437.2021.00792.
- **Primary URL：** <https://openaccess.thecvf.com/content/CVPR2021/html/Wertheimer_Few-Shot_Classification_With_Feature_Map_Reconstruction_Networks_CVPR_2021_paper.html>
- **正式全文：** <https://openaccess.thecvf.com/content/CVPR2021/papers/Wertheimer_Few-Shot_Classification_With_Feature_Map_Reconstruction_Networks_CVPR_2021_paper.pdf>
- **书目核实：** <https://api.crossref.org/works/10.1109/CVPR46437.2021.00792>
- **实际阅读范围：** CVF正式PDF第3-4页，Secs. 3.1-3.6、Fig. 2、Eqs. (1)-(13)，核对闭式重构、误差分类和辅助损失；CVF BibTeX。
- **作用位置：** 以每类支持局部描述矩阵通过带岭正则的闭式线性组合重构查询特征图，按重构误差形成类别logit；没有先压成单个全局支持向量。
- **核对到的方法：** FRN保留局部描述并按类别计算重构，但不要求预先指定逐位置语义对应。回归目标为查询特征图，不是类别标签；其重构结果用于误差评分。
- **与当前主稿的差异：** M3聚合版不以重构整幅查询特征为目标，而保留原传输的边际及参考权重，形成局部类距离并回写中间类别证据。
- **可入稿比较（1-2句）：** FRN通过类内支持局部特征对查询特征图进行岭回归重构，并以重构误差形成类别分数。本文M3采用保留传输边际的参考值聚合，将局部类距离转为空间类别证据回写，因此不以精细值对应或整图重构增益作为当前主张。
- **建议放置：** 2.3 与DeepEMD并列的局部参考组织比较；4.4.2 保留边际聚合；4.4.3 类别证据场。
- **写作边界：** 不能因未使用逐点硬对应，就把FRN称为无局部关系；本文保留边际也没有删除Sinkhorn和参考选择。
- **书目差异说明：** CVF BibTeX与PDF为8012-8021；Crossref登记8008-8017。采用CVF正式版本页码。

### A26 RSC

- **完整书目：** Z. Huang, H. Wang, E. P. Xing, and D. Huang, "Self-challenging Improves Cross-Domain Generalization," in Computer Vision - ECCV 2020, Lecture Notes in Computer Science, vol. 12347, 2020, pp. 124-140, doi: 10.1007/978-3-030-58536-5_8.
- **Primary URL：** <https://link.springer.com/chapter/10.1007/978-3-030-58536-5_8>
- **正式全文：** <https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470120.pdf>
- **书目核实：** <https://link.springer.com/chapter/10.1007/978-3-030-58536-5_8>
- **实际阅读范围：** ECVA正式原文第3-6页，Sec. 3.1、Eqs. (1)-(5)、Algorithm 1；Springer作者、卷号、页码与DOI。
- **作用位置：** 训练时对隐藏表示计算真实类logit的梯度，屏蔽梯度较高的一部分表示，再在屏蔽后的前向结果上计算分类损失更新模型。
- **核对到的方法：** RSC是根据梯度定位并抑制当前主导特征的训练策略，不是只展示显著图，也不是在固定模型上做输入敏感性评价。
- **与当前主稿的差异：** 主稿干预流程先对极化、区域和尺度做限定输入操作及均值控制，按实际分差改善筛选样本；随后保护原图教师类别优势下界，不等同于逐步删除高梯度特征。
- **可入稿比较（1-2句）：** RSC在训练中抑制对真实类预测梯度较高的特征，使模型利用其余表示。本文依据限定输入干预及均值对照筛选类别优势保护对象，关注何种操作确有判别收益，而不直接把主导响应统一视为应删除的信息。
- **建议放置：** 2.4 受控扰动与解释约束；4.6 分阶段优化及干预指导约束。
- **写作边界：** 干预训练结果属于主稿所述历史载体；不得因此宣称当前M1/M2/M3已完成同一干预适配或跨域收益。

### A27 Input-gradient Regularization

- **完整书目：** A. S. Ross and F. Doshi-Velez, "Improving the Adversarial Robustness and Interpretability of Deep Neural Networks by Regularizing Their Input Gradients," in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 32, no. 1, 2018, pp. 1660-1669, doi: 10.1609/aaai.v32i1.11504.
- **Primary URL：** <https://ojs.aaai.org/index.php/AAAI/article/view/11504>
- **正式全文：** <https://ojs.aaai.org/index.php/AAAI/article/download/11504/11363>
- **实际补充精读全文：** <https://arxiv.org/pdf/1711.09404>
- **书目核实：** <https://api.crossref.org/works/10.1609/aaai.v32i1.11504>
- **实际阅读范围：** 作者arXiv原文第2-3页，Input Gradient Regularization小节及式(6)、训练/评价设置；首面核实完整作者名。Crossref核实题名、venue、32(1)、年份及DOI；AAAI PDF搜索索引首末页核实1660与1669。
- **作用位置：** 在训练目标中加入交叉熵对输入梯度的平方范数，以双重反向传播更新参数；并非在网络中插入一个空间注意模块。
- **核对到的方法：** 核心目标为H(y,p)+lambda||grad_x H(y,p)||_2^2。它约束无穷小局部敏感性；作者亦区分不同梯度目标和正则强度。
- **与当前主稿的差异：** 本文对指定图像操作观察有限幅度类别竞争变化，以通过干预和均值控制的源样本构造教师分差下界保护；不直接最小化全部输入方向梯度。
- **可入稿比较（1-2句）：** Ross和Doshi-Velez通过惩罚分类损失的输入梯度，将局部敏感性约束纳入训练。本文按具体极化、区域与尺度操作的判别效应筛选优势保护对象，约束的是教师类别分差下界，二者作用于不同的训练量。
- **建议放置：** 2.4 紧接已有Right for the Right Reasons；4.6 干预指导约束的目标说明。
- **写作边界：** 这是2018年AAAI论文，不是已有refs[11]的2017年IJCAI论文；本文任务不转为对抗安全研究。
- **访问与阅读限制：** AAAI落地页/PDF正文访问返回拒绝，未声称从该站全文精读；方法核对使用作者arXiv全文，出版信息采用Crossref和AAAI原PDF的搜索索引。

## 建议整合顺序

1. 在2.2补入“类别响应与统计反馈”及“区域汇总和空间回写”两个短段，引用A15-A21。SCRC-Net仍用于船体结构采样对照；R-FCN用于相对位置读出基础，两者职责不同。
2. 在2.3以全局原型A22、局部到类A23、交叉注意A24、重构A25形成层次，再连接既有DeepEMD。收束到“参考选择、值聚合、类别证据及回写位置”，不把复杂传输自动解释为必要贡献。
3. 在2.4将A26和A27接到已有解释训练文献后；4.6只需要少量交叉引用，避免相关研究与方法重复。
4. 讨论中保留现有结论边界：M1独立交叉统计收益未确证、M1引导区域读出有配套收益、M3扩围结果混合、当前组合尚未重做历史干预适配。

## 元数据与访问说明

- 书目核查已使用web检索并读取primary原文；Crossref只补充出版元数据，未将其摘要当作方法证据。
- SE采用CVPR 2018三作者原版；不混入2020年TPAMI扩展版。
- GCNet为ICCV **Workshops** 2019。
- GSoP、DN4、FRN的Crossref页码与CVF不同，采用CVF正式PDF/官网页码，原登记值保留在JSON的alternate_pages。
- A2-Nets采用NeurIPS官方Metadata.json的352-361；CAN采用同类官方字段4003-4014。其他索引可能不同，整稿应一致采用本次明确指定的版本。
- 四篇NeurIPS论文未从官方记录获得出版DOI，JSON填null；不将arXiv编号或未经核实的数据库标识拼成DOI。
- A27的AAAI网页/PDF正文直访受限。方法已读作者arXiv全文；书目由Crossref核实，1660和1669可在AAAI原PDF的web索引核对。该访问限制不影响本次所用方法公式的核查。
- 本轮未新增与研究主题无关的工具/skill引用，未下载PDF到写作目录；仅交付Markdown和JSON。

