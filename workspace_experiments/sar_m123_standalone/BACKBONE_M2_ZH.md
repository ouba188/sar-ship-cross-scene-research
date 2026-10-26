# 实验性骨干内M2：区域池化—反馈块

此入口与现有M1/M2/M3末端系统并存，不覆盖其权重、CLI或推理逻辑。当前只实现M2骨干内版本，M1和M3尚未内部化。

## 模块与接口

`sar_m123.backbone_regions.RegionFeedback`接收`x[B,C,H,W]`、`masks[B,3,h,w]`、`valid[B,1,h,w]`，输出同形状特征。区域顺序为body/end/side。掩膜必须已经裁到有效原图范围，块会面积降采样并限制到有效范围。

1. 在每个空间位置沿通道LayerNorm。
2. 每个区域分别加权池化一个token，不把船体和外围合并。
3. 当前位置的query与区域token的key点积产生sigmoid反馈门。
4. 区域专属value变换后，用门和区域掩膜送回原区域；重叠区域归一化，区域外不更新。
5. 1x1输出投影后残差加入原特征。输出投影初始为零，gamma初始0.1，初始网络保持基线。

该块不是可变形卷积，也没有声称恢复真实散射机制；是可学习的区域池化和局部条件化反馈。核形状/OBB软区域沿用现有设计，不新增雷达方向条件。

## 嵌入与复用

```python
from sar_m123.backbone_regions import RegionResNet50

model = RegionResNet50(
    "E:/临时会话/sar_m123_standalone/assets/backbone/resnet50.pth",
    classes=8,
    stages=(1, 3),  # 两个独立实例；不要求同时存在M1/M3
    mode="regional",
)
# images: [B,2,1,H,W], float in [0,1], VV/VH分别预处理
# masks: [B,2,3,H,W]; valid: [B,2,1,H,W]; available: [B,2]
scores = model(images, masks, valid, available)
```

不同stage、不同极化使用独立参数；R50共享且参数和BN冻结，后续层梯度仍会传回适配器。没有把VV与VH像素直接相加。最终分别池化，再拼接两极化向量及可用性标记分类。

支持`mode='none'`基线、`mode='merged'`同参数量的区域合并对照。后者复用相同三区域的联合范围，不把额外背景作为对照差异。

冻结前缀可以用`prefix(stop=2)`缓存，再调用`forward_from(start=2)`训练后插入的块。若stage1已经有可训练块，禁止缓存经过该块的stage2前缀。完整`forward`支持任意stage1..4组合。

## 本轮验证边界

本轮训练仅stage2单点、单方向原尺度输入，不使用C4、M1、M3或参考库。三种子、无模块/区域合并/区域分离对照。全路径等价、梯度、BN冻结和重复放置有工程测试，不意味着多点放置已经取得分类收益。

不能把该小验证与原C4完整模型直接作公平强弱比较。先检验内部反馈是否有效，再决定是否接入完整管线和扩围。
