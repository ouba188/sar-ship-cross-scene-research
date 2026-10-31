# 当前完整模型：空间M1-M3 + layer2中段M2

2026-10-31，按用户决定将 `layer2[2]` 残差相加前的空间M2并入当前模型。
架构标识：`native160_layer2block2_preadd_spatial_m123`。
类入口：`sar_m123.current.CurrentModel`，具体实现 `sar_m123/current/spatial_layer2.py`。

## 当前流程

```text
VV/VH原生chip -> OBB中心裁剪/补边160，不缩放
  -> 共享ResNet-50 Stem/layer1
  -> layer2[0]/[1]
  -> layer2[2]: conv1 -> conv2 -> conv3+BN3 -> 空间M2 -> identity相加/ReLU
  -> layer2[3]
  -> layer3[0..4]
  -> layer3[5]: conv1+BN/ReLU -> M1 -> 空间M2 -> conv2 -> conv3 -> identity相加/ReLU
  -> 空间M3查询-参考匹配/回写 -> layer4 -> 有效像素全局读出
  + 带新M2的layer2区域读出 + M3后layer3区域读出/投影
  -> 拼接两极化全局与区域描述及可用性标记 -> 统一线性分类头
```

每极化：M1一次、内部M2回写两次、多层M2读出两处、M3一次。VV/VH共享骨干卷积，但模块参数独立；两处M2回写也不共享参数。没有5%交接或固定50/50类别分数融合。

M1/M2不含频域选择，M3的显式频带比较被屏蔽。当前数据接口仍保留旧频率字段及部分冗余描述计算，但这些字段不参与本模型输出。训练时细/中尺度的图像干预继续保留。

## 接入与训练

缓存边界前移到 `layer1`，形状 `[B,2,256,40,40]`。旧的整层layer2缓存不能复用。查询和参考的新增M2、下游卷积、参考value路径每次实时重算；冻结卷积仍向新增M2传递梯度。

M3比较依据继续来自不经过新增M2的冻结layer2/layer3原始路径；当前更新后的特征负责value回写与最终分类。因此新增M2没有同时改变比较锚点。

- `fit`：训练全部模块（含新增layer2 M2）、分类头和layer3[5]三个卷积。
- `select`：源标签指导原图/局部尺度干预/均值对照筛选，生成u和原图分差下界b。
- `adapt`：仍只更新layer3[5]三个卷积；使用所有源样本CE、原先判对样本KL和选中样本原图分差保护。新增M2应先通过fit学习，或从已有候选导入训练权重。
- `predict`：只需原始chip、OBB与固定源参考库，不需目标标签或干预图。

## 权重与使用

当前独立目录：`E:/临时会话/sar_m123_current_20261031`。
`weights/current.pt` 是空间M1-M3 s17权重与既存layer2[2] s17区域变换权重的组合；旧候选频域池化权重不导入。采用预先指定的同一训练顺序17，未按新队列分数择优。

此权重为 `integrated_unvalidated`，能够完整预测，新增M2不是零占位；尚未对该组合进行完整配套重训和精度评价。历史位置收益不能当作新组合结果。工程检查中的单步fit/adapt仅验证训练接口，其输出不替换current.pt。

在该独立目录运行：

```powershell
python -m sar_m123.current predict --model weights/current.pt --manifest manifests/demo.json --references manifests/references.json --output outputs/demo.json --device cuda
python -m sar_m123.current fit --model weights/current.pt --manifest manifests/source_fit.json --references manifests/references.json --output outputs/source_fit --seed 17
python -m sar_m123.current select --model outputs/source_fit/model.pt --manifest manifests/source_adapt.json --references manifests/references.json --output outputs/selection.json
python -m sar_m123.current adapt --model outputs/source_fit/model.pt --manifest manifests/source_adapt.json --references manifests/references.json --selection outputs/selection.json --output outputs/adapt_s17 --seed 17
```

原图、参考图、输入清单和完整检查点沿用现有格式。选样必须重新基于当前教师计算，不能套用旧模型的选样文件。

集成工具：`tools/integrate_layer2_m2.py`。省略 `--layer2-state` 时只增加零初始化M2，适合做结构一致性检查或重新fit；提供旧中段检查点时只迁移空间区域变换参数。

具体工程检查结果见独立目录 `verification/ENGINEERING_RECEIPT.json`；本文件不将实现验证写成分类增益。
