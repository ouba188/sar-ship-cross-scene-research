# 原生空间—频率联合矩提取

2026-10-24：已实现并实际运行完成，session `56039`，退出码 0。全部 1805 行已保存到本目录 `features.npz`，总耗时 **597.288 秒**，提取循环含首批对齐检查 **588.790 秒**。精确参数和检查数值见 `extraction.json`。

本次只执行特征提取，没有训练、预测、替代特征、调参或 Git 操作。缓存仅留本地；主任务独立处理训练、协议和分类结果分析。

## 输出契约

| 键 | 形状 | 类型 | 含义 |
|---|---|---|---|
| `joint` | `[1805,2,20,96]` | float32 | 每次旋转内计算 C，再做 C4 平均 |
| `marginal` | `[1805,2,20,96]` | float32 | 每次旋转内计算 M，再做 C4 平均 |
| `mean` | `[1805,2,20,32]` | float32 | 未单位化投影区域均值的 C4 平均 |
| `mass` | `[1805,2,20]` | float32 | 原始 mask 像素质量，沿用旧缓存；无效槽位为零 |
| `ids` | `[1805]` | int64 | 原 3387 行表的 `table_index` |

顺序严格对应 `../paired_scale_readout/table.csv`，不重新排序。样本组成：utility_train 644、selection 612、evaluation 485、strategy_fit 64；有效极化视图 3399。极化顺序 VV、VH；区域顺序 8 body、8 side、4 end。96 维按波长 4/8/16 优先，每个波长内按现有 PCA 的 32 个分量排列。所有输出有限，4224 个无效槽位的全部特征和质量为零。

## 计算定义

直接复用 `extract_tokens.py` 的原生输入、20 槽位和 `cell_mass(mask,4)`，以及 `frequency_conditioned_matching/extract_frequency.py` 的 Gabor `kernel` / `response`。原始 PNG 转 L 后除以 255；在完整原生图上反射延拓并滤波，然后按旧 placement 裁剪/填充到 160×160，不缩放。

每尺度采用固定图像方向 0/45/90/135 度，在每个原生像素先取四方向响应模平方的均值，再计算：

`e_b = log1p(mean_direction(abs(response_b)**2) / 1e-4)`

内容来自冻结、eval 模式的 R50 layer1。使用既有 `projection_layer1.npz` 的 mean256 和 components32×256，逐格计算 `F=(layer1-mean256) @ components.T`，不做逐位置或输出单位化。图像、能量图和区域 mask 同步旋转；每次旋转内计算：

```text
W = cell_mass(mask, 4)
V_b = cell_mass(mask * e_b, 4)
D = sum_grid(W)
Fbar = sum_grid(W * F) / D
E[e_b] = sum_grid(V_b) / D
E[e_b F] = sum_grid(V_b * F) / D
C_b = E[e_b F] - E[e_b] * Fbar
M_b = E[e_b] * Fbar
```

三尺度共享真实 mask 质量 D，不按能量质量归一化。积分采用原 stride-4 Voronoi 边界，保留半像素分数覆盖和边缘非等宽格，不用普通 4×4 分块替代。先计算各旋转的 C、M、Fbar，最后分别平均四次结果；不能把先平均的内容与能量重新相乘当作联合矩。

CNN 与逐格投影使用 float32，积分、矩和 C4 累加使用 float64，最终存为 float32。TF32 关闭，CPU 线程数 2，每批 8 行，GPU 为 NVIDIA GeForce RTX 5050 Laptop GPU。

## 定向验证结果

| 检查 | 最大误差/结果 |
|---|---|
| 旧 `tokens.npz:layer1` 投影均值重放，仅 `valid & mass>0` 的 67976 个槽位 | **1.1444091796875e-5**，预设容差 2e-4 |
| 常量能量下 C=0 | 3.6637359812630166e-15 |
| 内容、mask 权重及能量权重同步网格置换 | 1.1102230246251565e-15 |
| 独立像素/网格重叠积分参考 | 1.5543122344752192e-15 |
| 分数网格总质量守恒 | 7.105427357601002e-15 |
| 仅置换内容的辨别性对照 | M 不变，C 最大改变 2.0 |
| 首批 15 个真实视图同步旋转，完整 C4 重新提取 | C 1.77636e-15，M 2.84217e-14，Fbar 1.77636e-15 |
| 四方向平均 Gabor 能量图旋转 | 1.2434497875801753e-14 |
| 新 mask 与旧缓存质量相对误差 | 5.9255452537842974e-8 |
| 原生像素 placement 对齐 | 0 |
| 有限值、无效槽位零值、保存后逐数组读取一致性 | 全部通过 |

代数检查固定 seed=1024。C4 检查验证联合变换与四次汇总的对齐，不假设单次 CNN 特征具有旋转等变性。本次没有分类预测，因此不产生判对/判错、救回/误伤或分类收益结论；相关分析由主训练任务处理。滤波支撑与 CNN 感受野仍可跨越区域边界。

## 复现命令

在 `E:/临时会话` 执行：

```powershell
& 'D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe' -X utf8 -B 'scene_aware_classification_20261018/model_closeout_20261023/local_evidence_transport/spatial_frequency_joint_pooling/extract_joint.py'
```

该命令重新计算并覆盖本目录 `features.npz` 和 `extraction.json`；不启动训练。
