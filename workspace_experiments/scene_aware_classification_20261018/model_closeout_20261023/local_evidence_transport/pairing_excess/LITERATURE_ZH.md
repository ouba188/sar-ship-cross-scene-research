# 相关方法与差异边界

2026-10-24。普通web工具本轮未返回可读正文，后使用search_service读取作者arXiv全文HTML和官方代码；以下仅按实际可读来源说明，不声称读取失败的PDF或网页。

## DeepBDC，CVPR2022

已读arXiv方法3.1-3.3及官方bdc_module.py。该方法将Brownian distance covariance用于少样本分类，通过特征距离矩阵双中心化构建图像表示，再与类别原型比较。官方实现沿通道构建矩阵。因此，依赖建模、双中心化、联合与边缘分布比较均非我们首创。本轮covariance臂只是简单二阶替代，并非DeepBDC全模型复现。

- https://arxiv.org/html/2204.04567
- https://raw.githubusercontent.com/Fei-Long121/DeepBDC/main/methods/bdc_module.py

## 项目中已读的直接邻近工作

RENet在局部相关张量上学习并聚合，IFR利用支持空间特征重构查询；读出与访问边界沿用../learned_edge_metric/ALGORITHM_POSITION_ZH.md。SCRC-Net结构化采样与我们的目标区域组织比较见早期OBB实验记录。不将这些论文描述成只做全局平均。

## 本轮准备检验的区别

对象是OBB内局部响应与其外侧/轴端响应之间的绑定，而非一般图像通道依赖。对同一查询和同一参考，先构造所有局部重配的匹配矩阵，再以真实绑定相对等内容重配的超额值形成类别证据。与原mixed的b-n不同，不先丢失端点内容；与joint不同，去掉可由各端独立内容解释的匹配部分。它仍是已有依赖建模思想上的任务化推进，只有配对对照与完整分类证明价值后才支持组件创新，暂不声明领域首创或顶刊已达标。
