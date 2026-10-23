# 手工信息注入CNN：实际方法和任务边界

Tianwen Zhang, Xiaoling Zhang. Injection of Traditional Hand-Crafted Features into Modern CNN-Based Models for SAR Ship Classification: What, Why, Where, and How. Remote Sensing 13(11):2091, 2021. DOI 10.3390/rs13112091。

官方PDF：https://mdpi-res.com/d_attachment/remotesensing/remotesensing-13-02091/article_deploy/remotesensing-13-02091.pdf 。requests+pypdf实际读取；已读方法与实验设置至第21页，第1至10页初次工具输出少量截断，第11至21页完整文本输出。未精读第22至32页结果/讨论，因此不把摘要最大增益当作核验过的公平性能结论。

## 核心做法

从图像额外提取HOG、几何、局部RCS或主轴描述，与AlexNet/VGG/ResNet/DenseNet分类特征融合。作者比较拼接、加权拼接、映射后加法与拼接等，推荐手工特征L2归一化、MLP映射至对应维度，再加权拼接进入末端分类。原文提出这些设计的动机，不等于已证明所有不同物理含义的向量不可相加，或BN与L2具有相同机制。

## 与当前数据直接相关的限制

第21页明确说明：其OpenSARShip实验用1.0的SLC而非我们当前GRDH，三类、每类338训练，其余测试；没有报告我们按过境与身份隔离的设置。论文还明确因2.0背景复杂、自动提取最小船框困难，未将2.0用于该实验。

因此，可借鉴“已有抽象特征之外补充受控描述”，但不能直接移植其结果数字。当前研究的可能推进是：在原尺度GRDH及复杂背景下，不依赖可靠首尾或精确船体分割，在已有类别证据坐标中组织局部关系，并与幅度、几何支持量和普通附加非线性分开验证。

这只是待验证的任务与方法差异，不能宣称第一次融合先验/局部统计/手工特征。若关系没有独立收益，就不能仅凭OSS2任务更难将其包装成有效模块。
