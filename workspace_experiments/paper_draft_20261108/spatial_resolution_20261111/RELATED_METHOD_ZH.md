# 相近设计依据

## Dilated Residual Networks

Yu, Koltun, Funkhouser, CVPR 2017。原文：https://arxiv.org/html/1705.09914 。本轮阅读第2节转换公式、第4节网格伪影、第5.1节和表1。

借鉴：删除部分下采样，同时增大后续卷积膨胀率，避免将空间网格增加与感受野缩小混为一项改变。原文在ImageNet分类比较相同深度与参数量的ResNet/DRN-A，另以DRN-B/C区分附加去网格设计；并非只展示定位热图。

本轮只移除layer3入口的一次下采样，不增加去网格模块。与论文的从头配套训练不同，我们采用已训模型短继续训练，因此不将阴性结果扩展为对密集骨干的普遍否定，也不把正结果当独立创新。

## 官方实现

https://docs.pytorch.org/vision/stable/_modules/torchvision/models/resnet.html

核对Bottleneck与_make_layer：首个block使用previous_dilation，后续block用更新后的dilation。移除layer3步长时同时修改残差投影支路步长；layer4保留其步长但使用累积膨胀率。本轮数值检查原采样格点与密集网格的偶数位置吻合，避免两支路或坐标错配。

这些资料支持实现逻辑，不提供本任务的涨点保证。当前M3比较描述已包含浅层信息；本轮主要改变深层区域处理、回写和最终读出，不能称为首次引入高分辨率信息。
