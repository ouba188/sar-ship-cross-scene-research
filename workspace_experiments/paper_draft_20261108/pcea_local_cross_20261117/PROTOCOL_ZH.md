# 双极化局部内容交互的配套验证

2026-11-17，训练前固定。候选实验，不改正式模型/论文。

## 问题与已有证据

上一轮local/both是本极化20->10局部特征与对侧区域摘要交互，不是local-to-local；选中模型228条预测与旧PCEA相同。新一轮检验两路局部内容保留到交互之前是否增加分类效用，不预设其为主要根因。

旧原图记录000955、003656显示两极化局部亮峰分布不同，但两例均未被旧方案救回；这只是可用内容差异，不是必然判别价值。原生区域有效格数在20约5，降10约2，但粗writer也存在稳定判对样本。VV/VH不假定像素对应；仅同一检测目标的对应区域内做内容参考，不宣称电磁散射点配准。

## 固定协议

- OSS2既有719训练/228开发验证/32源参考；验证206双有效、22单有效。既有验证集已经反复使用，不能称独立测试。
- 原始输入有效性及OBB不变，不混入恢复缺失极化的收益。原Full/分类头冻结；CCREL投影与PCEA联合训练。正式Full曾做全骨干微调，本轮冻结只是诊断条件。
- A旧pooled affine_frozen两档LR复用已有完整记录。B/C/D各2LR，每次4epoch/120step，seed17；相同初始化、增强、损失、优化器、数据顺序、可训练参数和writer上界。reader1e-5；exchange1e-3或3e-4；head冻结，继承head字段不生效。
- 选优BA/OA/-NLL，epoch0可选；不靠测试选温度/配置。新增cross_local用cosine attention温度0.25，无额外QKV矩阵，沿用32->16投影；这是本轮固定候选，不是最优性声明。

## 配套组别

|组别|自己的局部内容|参考内容|比较目的|
|---|---|---|---|
|A pooled|区域摘要|对侧摘要|已有基准，复用|
|B independent|20x20|本极化局部投影的区域均值|局部加工，不跨极化|
|C cross_mean|20x20|另一极化局部投影的区域均值|对侧整体条件|
|D cross_local|20x20|另一极化同区域全部有效局部投影，内容注意力|对侧细粒度内容选择|

B/C/D同样保留均值+4类别槽；局部残差在当前CCREL源单位下中心化、限幅0.5后加到各槽，32->16投影+tanh。D用当前本极化局部作为query、另一极化局部作为key/value；soft区域面积是key先验，不输入目标标签。五槽分别匹配，没有硬top2过滤。B/C使用相同局部token集合的面积加权均值，不把非线性投影前后均值混为一谈。

所有组在20x20生成门控联合特征，再按origin=0的20/10网格真实单元边界重叠及本区域占比降10x10，使用相同80->256 scale/bias矩阵，回写layer3[5].conv2输入；保持0.1区域RMS上界。没有扩大writer分辨率或新增层。该重采样把fine格内容/区域占比作分片常数近似，不宣称恢复格内部真实散射位置。

### 开始训练后的坐标修正

原先普通2x2区域池化与既有origin0 writer的网格中心不一致；原图/几何侧审阅发现000955 VH body约9.81%的writer mask权重对应到空的2x2 reader组。没有根据分类结果调这项；这是可达的坐标映射问题。停止时旧草稿四个完整fits及D/base中断fit已产生，原模型代码/协议和记录保存在draft_pool2，不覆盖、不作为正式B/C/D结论。修正后B/C/D统一从同一原Full重启六次拟合。原图、OBB、writer本身均不修改。

## 验证及决策

1. 算子检查：SDPA与显式小矩阵结果/梯度一致，区域外key不能影响输出；uniform为面积加权均值；常量对侧退化为同值。检查初始重放、训练梯度和单极化不改动。
2. D与独立训练B/C比较，不把仅超A当交互价值。同权重D的uniform、other_mean、disabled诊断用于解释，不代替重训对照。均值化发生在局部非线性投影后，隔离已形成的局部token差异。
3. 若D开发分类净增益优于B/C且不只是NLL变化，再冻结配置补29/43并在378历史评价确认；否则保存失败边界，不自动扩到24港或做无依据搜索。OA/BA分歧按类别与错例解释，不要求全类别全场景胜。
4. 每轮查看原始VV/VH、实际合法同类/混淆类参考、持续对/错、救回/误伤。预指定000955/003656并非只挑正例；有新翻转再核查是否依赖局部选择。不将注意力当物理因果证据。

## 参考范围

CrossViT (ICCV2021, Chen et al.)提出跨分支token交互；本轮不照搬CLS汇总，而检验双极化区域内局部内容参考。它不是本任务收益证据。原文：https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_CrossViT_Cross-Attention_Multi-Scale_Vision_Transformer_for_Image_Classification_ICCV_2021_paper.pdf

SDPA使用官方API，明确soft区域mask、dropout=0和scale=4；小矩阵对照检查实际运算而非只看接口名：https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
