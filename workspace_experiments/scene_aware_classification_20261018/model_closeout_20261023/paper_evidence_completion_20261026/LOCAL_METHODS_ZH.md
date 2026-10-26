# 冻结局部特征基线算子

本文件记录独立算子与接口，不含 runner、训练、目标集选型或准确率结果。
实现只依赖 NumPy、PyTorch；可选 EMD 另依赖当前已有的 SciPy。

## 文献核对与复现边界

2026-10-26 在线核对 CVF 官方论文全文，直接读取 PDF 第 3、4 页（印刷页 7262、7263）。
搜索工具直接打开 PDF 返回 403，随后通过 Python urllib 成功读取同一官方 PDF，
在内存中用 PyMuPDF 提取正文；没有下载文件到允许范围外。

Wenbin Li, Lei Wang, Jinglin Xu, Jing Huo, Yang Gao, Jiebo Luo.
*Revisiting Local Descriptor Based Image-To-Class Measure for Few-Shot Learning*.
CVPR 2019, pp. 7260–7268.

- 官方页面：https://openaccess.thecvf.com/content_CVPR_2019/html/Li_Revisiting_Local_Descriptor_Based_Image-To-Class_Measure_for_Few-Shot_Learning_CVPR_2019_paper.html
- 官方全文：https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Revisiting_Local_Descriptor_Based_Image-To-Class_Measure_for_Few-Shot_Learning_CVPR_2019_paper.pdf
- 作者代码来源：https://github.com/WenbinLee/DN4 （本次未复制作者代码）

第 3.3 节式 (2) 定义：对每个查询局部描述符，在整个支持类的描述符池中寻找 k 个
最近邻，再对所有查询描述符及其 k 个邻居的余弦相似度求和。
第 3.2 节说明类内跨图像合并描述符；第 3.3–3.4 节说明 embedding 与匹配模块
通过 episodic training 联合训练。因此此处是 **DN4 核心匹配算子在冻结 SAR 特征上的复现**，
不是原论文端到端训练、骨干或 benchmark 复现。

本任务固定 k=3。相对于式 (2)，额外对每个极化的有效查询 token 数取平均，
然后对可用极化等权平均；这是本项目缺失观测适配，必须在论文中说明。
仅在查询 token 数固定且极化资源一致时，该归一化才对应统一缩放。

## 输入与共用约定

已阅读 `oss1_m123_replication_20261026/train_m3.py` 的 `Matcher`、`prepare_local`
及 `sar_m123_standalone/sar_m123/tokens.py`。
使用已有投影后的 `x`，不拟合或改变特征；输入可为 NumPy 或 torch：

```python
query       # [N, 2, 20, D], p=0 VV, p=1 VH，当前投影 D=32
query_valid # [N, 2, 20], 用 mass > 0，不是 tokens.extract 的 [N,2] valid
bank        # [B, 2, 20, D]，源参考 32 个，每类 8 个，全部传入
bank_valid  # [B, 2, 20], bank_mass > 0
bank_y      # [B], 整数标签 0..C-1；输出列就是 class_id
```

三区域顺序为 `[0:8]` 船体、`[8:16]` 侧部、`[16:20]` 端部。
算子会再次单位化 token；mask 为假、含 NaN/Inf、范数非有限或 <=1e-8 的 token
均从计算与计数中剔除，不允许零向量占据最近邻名额。
不使用像素 mass 大小加权，也不先筛选最近的两张参考图。

`query_ids`、`bank_ids` 必须同时提供，长度分别为 N/B；相等 ID 的所有参考图被排除。
为排除同船，应传相同规范的船只 ID/MMSI，而非 obs_id。函数不转换字符串/整数 ID。
runner 负责源角色合法性与选择不含 bank 角色的实测 fit 样本。
`fit_only` 与其他源标签组织对照由 runner 确定，算子不读取角色、源/目标标签划分或结果；
不会根据其他 baseline 的准确率改变 k、区域方式或参考池规则。
`num_classes` 默认 `max(bank_y)+1`，空 bank 时须显式提供。

输出默认 `[N,C]`；`return_available=True` 返回 `(scores, available)`，后者为同形布尔矩阵。
NumPy query 返回 NumPy；torch query 返回相同设备的 detached torch，无梯度。
float64 query 保留 float64，其余浮点计算使用 float32。
无有效查询极化、参考全被排除或某类无证据时，返回该项零分及 `available=False`。
**零分是无证据哨兵，不代表真实中等相似度**：如果做独立 argmax，runner 应屏蔽
无证据类别，并对整行无证据进行明确回退；若作为残差特征，可结合 availability 使用。

## 函数接口

```python
dn4_scores(query, query_valid, bank, bank_valid, bank_y, *,
           k=3, query_ids=None, bank_ids=None, num_classes=None,
           batch_size=128, return_available=False)

regional_prototype_scores(query, query_valid, bank, bank_valid, bank_y, *,
                          merge_regions=False, query_ids=None, bank_ids=None,
                          num_classes=None, batch_size=128, return_available=False)

balanced_emd_scores(query, query_valid, bank, bank_valid, bank_y, *,
                    query_ids=None, bank_ids=None, num_classes=None,
                    batch_size=128, return_available=False)
```

### DN4

每类、每极化合并所有参考图的有效 token。每个有效查询 token 与该池求余弦，
取 top-k **求和，不除以 k**；再平均有效查询 token，最后平均可用极化。
正负余弦都保留。若排除后只有 1 或 2 个有效参考 token，则求实际可用邻居之和，
不补零邻居、不补重复邻居、不把缺失邻居计入求和；空池无证据。
这种不足 k 的退化约定并非原论文处理规则，正常每类 8 张参考图时通常不触发。
跨区域匹配不受区域边界限制，保持类池的 DN4 核心形式。

### 普通区域均值原型

每极化、每区域，把所有合法参考图的有效单位 token 等权平均，得到类均值，
查询在同区域作有效 token 均值，计算两均值的余弦。先等权平均可用区域，
再等权平均可用极化。均值抵消为零的区域不提供余弦证据。
这是有效 token 等权，不是每张参考图等权；有效 token 更多的图贡献更多。
`merge_regions=True` 时先合并 20 个 token 作单一均值；默认保留三区域。
两种形式事先确定，不通过 target 结果选择。

### 通用 balanced EMD

每个查询与每张参考图、每极化分别求解最多 20×20 的线性规划：
地面距离 `1-cos`，查询质量各为 `1/n`，参考质量各为 `1/m`，
非负运输矩阵的行/列边缘分别等于这两组均匀质量。
配对得分为 `1 - 最小运输代价`；每类每极化平均可用参考图得分，
再等权平均可用极化。所有参考图均参与，不作逐查询参考筛选。
SciPy `linprog(method='highs')` 在 CPU 逐对求解；`batch_size` 仅为接口一致而接受，
不是 LP 批量并行。输入 torch CUDA 时也回到 CPU 解 LP，再返回原设备。
只缓存不同 n/m 的边缘约束，不缓存数据或目标结果。

这是 **均匀质量的通用 OT 对照，不是完整 DeepEMD**，不复现其学习流程或专门组件。
LP 初测：当前环境 128 次 20×20 随机代价求解 0.3405 秒、全部成功。
其时间随查询×参考×极化线性增长；优先使用 DN4/区域原型，EMD 为可选项。

## 验证

仅使用确定性手工 toy 与固定种子合成 token，不读取 target 或任何实际分类结果。
执行 `python -B test_local_baselines.py -v`，`-B` 禁止生成范围外字节码文件。
覆盖手工 top-3 类池求和、最后几张参考图确实参与、mask/非有限/零向量、
不足 k、极化等权和单极化回退、三区域/合并原型、手工非等长均匀质量 EMD、
同船排除、无有效极化、空 query/bank、抵消为零的原型、参考顺序不变性、
批大小与 NumPy/torch 一致性。

`python -B test_local_baselines.py --benchmark` 在 8 个合成查询、32 张参考、
4 类各 8 张、2×20×32 特征上计时；不进行准确率比较或选择方案。
2026-10-26 实测：9 项 unittest 全部通过，测试主体 1.073 秒。
环境为 `D:/Program_files/anaconda3/python.exe`，NumPy 2.3.5、PyTorch 2.10.0+cu128、
SciPy 1.16.3；此次验证在 CPU 上运行，torch 线程数为 2，未验证 CUDA 执行路径。

| 算子 | 合成输入计时 | 输出 |
| --- | ---: | --- |
| DN4 | 0.0259 秒 | `[8,4]`，全部有限 |
| 三区域均值原型 | 0.0124 秒 | `[8,4]`，全部有限 |
| balanced EMD | 2.0666 秒 | `[8,4]`，全部有限 |

以上为单次调用墙钟时间，包含各函数首次调用的开销；EMD 包含 512 次 LP，
不能把最初随机代价 LP 微基准当作整个匹配流程的耗时。
未训练、未读取 target 结果、未作 Git 操作；只创建本目录三个指定文件。
