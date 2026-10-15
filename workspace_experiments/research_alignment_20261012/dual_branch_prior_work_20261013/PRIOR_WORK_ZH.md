# SAR 第二路设计的已有依据

日期: 2026-10-13。范围: 回答前人为何设置第二路及与当前研究的关系; 未运行新模型实验, 不是穷尽性 novelty 结论。

## 结论

第二路通常针对明确缺口: 额外极化观测、显式形状/纹理、不同网络表征、不同预训练经历, 或训练期的辅助监督。双分支、特征互补、贡献协调本身均有先例。最接近当前网络组合的是 DSAN++ 的 R50+R18, 应优先比较, 不能只引用距离较远的方法。

## 1. DSAN++: 深浅网络表征与跨域对齐

- Zhao, Lang, 2022, Improving Deep Subdomain Adaptation by Dual-Branch Network Embedding Attention Module for SAR Ship Classification. DOI: 10.1109/JSTARS.2022.3206753。
- 方法全文: `https://www.researchgate.net/publication/363595936_Improving_Deep_Subdomain_Adaptation_by_Dual-Branch_Network_Embedding_Attention_Module_for_SAR_Ship_Classification`
- R50 与 R18 并行, 池化特征拼接; CBAM 关注可迁移特征, LMMD 对齐类别条件子域。作者用深浅网络的语义/细节互补解释设计; 这是设计依据, 不是 R18 必然具备更高输出空间分辨率的定理。
- 任务为有标签光学源域到无标签 SAR 目标域, ImageNet 初始化后联合训练。不是当前冻结编码器、SAR 跨港口、少量目标支持标签的协议。
- 消融将 DBN 与 CBAM 分开。表 II/III 中仅增加 DBN, GF-SAR 准确率从 82.67 到 84.67, HR-SAR 从 84.00 到 85.11; 完整模型的 89.11/88.89 不能全部归于第二路。
- 同时报告分布距离、混淆矩阵和嵌入可视化; 不宜说前人只给总体准确率。

## 2. HOG-ShipCLSNet / MFCFNet: 显式视觉线索及贡献协调

- HOG-ShipCLSNet, TGRS 在线 2021 / 卷 60 (2022), DOI: 10.1109/TGRS.2021.3082759。
- 全文: `https://www.researchgate.net/publication/352106909_HOG-ShipCLSNet_A_Novel_Deep_Learning_Network_With_HOG_Feature_Fusion_for_SAR_Ship_Classification`
- CNN 表征之外加入同图 HOG, PCA 后融合。动机是利用梯度方向描述的形状/边缘, 不是引入新的传感器信息; 也不是两个可学习 CNN 编码器。
- MFCFNet, TGRS 61 (2023), 5212614, DOI: 10.1109/TGRS.2023.3297648。
- 全文: `https://eprints.lancs.ac.uk/199400/1/Multi_Feature_Collaborative_Fusion_Network_with_Deep_Supervision_for_SAR_Ship_Classification.pdf`
- DEEP 分支处理原图, HAND 分支处理 Canny/Harris/Gabor/LBP 等选定特征图。辅助分类监督及双向 KL 促进协同, 通道注意力同时调节融合与辅助损失贡献。
- 不仅考虑增加线索, 已明确考虑直接融合不足和两路贡献不平衡。消融见代理笔记; OpenSARShip 与 FUSAR 分别监督训练测试, 不是跨数据集迁移。
- 推理仍保留 HAND 编码器, 删除辅助分类头。普通图像可生成手工特征, 因而仅依赖图像输入不能作为我们的独有区别。

## 3. DPIG-Net: 利用额外极化观测

- Remote Sensing 2023, 15(8), 2138, DOI: 10.3390/rs15082138。
- 已读全文为 2022 预印本: `https://arxiv.org/pdf/2207.04639`; 期刊全文访问失败, 不将预印本数值冒充期刊最终结果。
- 实际三分支: VV 为主, VH 与跨极化乘积幅度图为辅; VV 引导交叉注意力, 融合后保留主分支全局残差。
- 动机是 VV 单独分类没有利用全部可用极化观测, 并避免辅助融合稀释主分支。评估为 OpenSARShip 内部划分。
- 不能称其为两个普通图像编码器, 也不能把所印公式的幅度乘积直接解释成独立相位信息。具体公式与数据要求的差别见 agent_notes.md。

## 4. DCPNet: 第二路作为预训练监督

- Dual-stream contrastive predictive network with joint handcrafted feature view for SAR ship classification, arXiv 2023 / ICASSP 2024。
- 全文: `https://arxiv.org/html/2311.15202v1`
- 自监督训练融合弱/强增强的一致性与手工特征预测。作者认为简单拼接冗余, 直接对齐差异较大的深度/手工嵌入也困难, 所以通过预测头转移知识。
- 下游使用分离出的编码器进行分类评估, 不等于部署时必须保留两个判别分支。附录比较直接对比与特征预测, 并消融手工特征任务。
- OpenSARShip/FUSAR 的各自数据集内评估, 不能当作本项目跨港口协议的直接结果。

## 5. 互补预训练: 以不同错误类型支持组合

- Boosting ship detection in SAR images with complementary pretraining techniques, arXiv:2103.08251 (2021)。
- 全文: `https://arxiv.org/html/2103.08251v1`
- 光学遥感船舶检测预训练 OSD 与光学-SAR 匹配预训练 OSM 分别产生 SAR 检测器, 再加权融合框。
- 作者观察 OSD 更利于海上召回, OSM 更利于减少陆地虚警, 据此解释组合; 有单路径、组合及近岸/离岸比较。
- 这是检测而非分类。不能直接比较分数, 也不能说每项指标都改善; 但说明不同预训练来源和错误互补并非全新动机。

## 对当前论文的具体含义

可发展的研究问题是: 通用监督预训练与 SAR 自监督表征, 在明确的跨港口/跨数据集与少量支持标签条件下, 是否保留当前单路未利用的任务判别能力, 何时救回、何时误伤, 如何据此利用第二路。

这只是候选贡献定位, 不是已证明预训练来源造成收益。架构、预训练来源、输入通道/处理、模型容量及适配规则仍有混杂, 需要对应替换对照才能归因。双分支或 R50+R18 组合本身不构成区别; 也不能把前人概括成只拼接、从不考虑协调。

本地证据边界: `E:/临时会话/branch_root_link_20261013/FINDINGS_ZH.md` 支持输入配套后仍有部分净收益和有限重建未覆盖的任务价值, 同时记录 SLC 收益抵消/负值。它不证明独占物理信息, 不证明任意强单路都无法替代, 不证明普遍鲁棒, 也尚未隔离预训练来源的因果作用。

细读 DPIG-Net/MFCFNet 的页码、消融和输入限制见同目录 `agent_notes.md`。本文未修改既有负结果或实验设置。
