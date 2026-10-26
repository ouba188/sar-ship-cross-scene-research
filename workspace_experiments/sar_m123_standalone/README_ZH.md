# M1–M3 独立 SAR 船舶分类工程

## 2026-10-26 新版入口

**新版与下文历史八分类工程并存，不能混用资源。** 新版是已保留的公开四分类 `raw_summary + adapted PairMetric + paired readout refit`，完整说明见 [ALIGNED_MODEL_ZH.md](ALIGNED_MODEL_ZH.md)。使用 `--model aligned --assets assets_aligned/oss1` 或 `assets_aligned/oss2`；默认 `--model formal` 保留历史八分类行为。

新版提供 chip 到分类的独立推理、原图训练特征提取、分阶段训练及模型资源安装入口。下文历史证据图册、八分类结果与接口默认值仍对应旧正式版本，不代表新版结果。新版验证记录见 `validation_reports/aligned*`。

本目录是已经接通的 **chip 图像 + 检测 OBB → 分类输出** 推理工程。它不是旧 `replay_selected.py` 的包装：输入图像现场提取特征，M1、M2、M3 在本目录内实现，运行时不导入旧实验脚本、不读取查询特征或查询分数缓存、不需要查询标签。

本机 `assets/` 已导出所选模型的全部推理资源。复制整个目录（包括 `assets/`）即可迁移；GitHub 只保存代码、文档和小型验证记录，不保存权重、源参考特征库或原图。

真实模型证据图册：`evidence_gallery_20261025/index.html`。含M1类别响应、M2实际区域、M3真实参考与匹配、分数变化及反例；图由现有检查点重新计算，无新增训练。绘图依赖可通过 `pip install -e ".[visualization]"` 安装。

研究逻辑与证据：[问题、受控干预、M1–M3算法和案例的完整对应](RESEARCH_EVIDENCE_CHAIN_ZH.md)。区分历史4744观测的组件对照与当前485开发查询的冻结版本，不把不同实验的收益相加。

## 1. 快速运行

本机已验证环境：Python + PyTorch 2.7.1/cu128、torchvision 0.22.1、NumPy 2.4.4、SciPy 1.17.1、Pillow 11.1.0。CPU 和 CUDA 均已实际运行。新环境安装工程依赖可执行 `python -m pip install -e .`；需要 CUDA 时先安装适合机器的 PyTorch。

本机真实图像示例已经准备好：

```powershell
Set-Location 'E:\临时会话\sar_m123_standalone'
& 'D:\Program_documents\Anaconda_envs\envs\ClearSAR\python.exe' -X utf8 -m sar_m123 --input validation_local/requests.json --output outputs/predictions.json --seeds 17 29 43 --trace
```

自己的输入使用相同命令，替换 `--input`。`examples/request_template.json` 是格式模板，不含真实图片。JSON 可以是一条记录，也可以是记录列表；相对图片路径相对于输入 JSON 所在目录。

一般只运行一个冻结配置，例如 `--seeds 29`。三个编号代表原实验的三个训练配置，**不是运行时随机种子，也不会自动集成**。默认17是接口默认值，不代表它最好。`--device cpu` 强制 CPU。安装后从其他位置运行时，请显式指定 `--assets` 为资源目录。

Python 接口：

```python
import json
from pathlib import Path
from sar_m123.pipeline import ShipClassifier

root = Path("E:/临时会话/sar_m123_standalone")
request_path = root / "validation_local/requests.json"
records = json.loads(request_path.read_text(encoding="utf-8"))
model = ShipClassifier(root / "assets", device="cpu")
result = model.predict(records[0], request_path.parent, seeds=(29,), trace=True)
print(result["predictions"][0]["class_name"])
```

## 2. 输入约定

```json
{
  "id": "query_001",
  "vv": {
    "image": "chips/query_001_vv.png",
    "target_index": 0,
    "detections": [
      {"corners": [[32,24],[48,40],[42,46],[26,30]], "confidence": 0.9}
    ]
  },
  "vh": null
}
```

| 字段 | 含义 |
|---|---|
| `vv` / `vh` | 两个极化分别给出；真正缺失的极化使用 `null` |
| `image` | 与训练预处理一致的8位显示 chip；本工程不把原始 SAR 浮点或16位幅值悄悄拉伸成训练输入 |
| `detections` | 本 chip 中的目标及其他检测 OBB。四角按边界顺序排列，坐标为 chip 像素，x向右、y向下 |
| `target_index` | 本次要分类的框在 detections 中的索引，从0开始；不是最大框或最亮框自动替代 |
| `confidence` | 检测器置信度，原协议阈值0.25；`shape_usable:false` 可明确排除不可用检测框 |
| `alignment_valid` | 可选，默认 true。跨极化关联无效时设 false，该路不进入模型 |
| `radar_axis_rad` | 可选，**已经变换到当前 chip 像素坐标系**的雷达轴弧度。不能直接把地理航向角或入射角填在这里 |
| `ship_axis` | 可选的已检测轴，包含 `center:[x,y]`、`length_px`、`width_px`、`angle`（弧度）；省略时从目标 OBB 推导，不使用 AIS 航向 |

至少一个关联有效的极化必须有可用目标框。图像存在但目标框不可用时，可以提供 `target_index:null` 或 `-1`：旧基础路径仍可读取图像背景，M1/M2/M3 不产生该极化的目标证据。这与“图像完全缺失”不同，也是本轮迁移中实际发现并修正的接入问题。

原生像素保持：大 chip 围绕目标裁边，小 chip 补零至160×160，**不缩放船体**；补零不算有效区域。目标本身超过160像素画布时明确报错，不偷偷截断或缩小。原数据的显示拉伸、像素间距与检测关联仍属于上游输入条件；不缩放不意味着不同传感器的每像素物理尺度自动相同。

缺少可靠雷达轴时沿用训练过的图像轴回退，不伪造雷达几何。这里的船轴是无向轴，不能解释为已知船首、船尾。

## 3. 实际计算链

```text
VV/VH PNG + 各自目标 OBB / 其他检测框 / 可选几何
  -> 原生裁边或补零，保留目标/其他检测/剩余输入区域
  -> VV、VH分别做4个90度旋转视图，分别经过同一个冻结R50
  -> stem、layer1、layer3特征
  -> 已有基础分类路径，得到8类分数 B
  -> M1：均值支路 + 双侧类别响应注意力，得到 S1
  -> M2：OBB区域身份保持池化 + 类别修正，得到 S12
  -> M3：局部空间—频域描述 -> 源参考检索 -> both比较与匹配
          -> 独立VV增量 / 零证据中心化VH增量
          -> B锚定的空间—频域条件化极化权重
  -> S = S12 + gVV * deltaVV + gVH * deltaVH
  -> argmax：8类船型
```

这里“B锚定”指历史选定的极化比例锚点，不是让基础分数 B 再重复加入一次。

### 基础路径

`baseline.py` 保留实际选定模型依赖的轴向区域特征、五点读出、单独邻域输入读出与校准头。不能把这部分替换成一个普通 GAP 分类器后仍声称是同一模型。邻域隔离输入需要额外经过共享 stem；四旋转也是真实的多次特征提取，不是旋转已缓存特征图。

### M1：类别响应注意力，保留均值支路

`m12.py` 的 `SummaryReader` 用冻结源端类别读出矩阵把 layer3 空间特征变成8类响应图。按有效输入范围计算类别响应均值，以及中心化响应的正、负两侧 softmax 加权摘要。`AttentionHead` 分开处理均值与局部分布，再输出相对于基础分数的类别增量。VV/VH先分别统计，不先混成一张响应图。

### M2：区域身份保持的类别证据池化

`regions.py` 根据 OBB 构造船体、轴端、侧向软区域；`m12.py` 在 layer1 类别响应图上分别读出均值与两侧摘要，同时保留覆盖量、有效单元数及区域有效标记。`RegionHead` 利用这些独立摘要生成增量，接在 M1 后。**当前正式版是池化读出，不是新训练的船轴卷积**。

### M3：空间—频域条件化证据匹配与融合

`tokens.py` 保留20个局部槽位的空间身份；`frequency.py` 计算三尺度、四方向 Gabor 响应及内容—响应联合描述。VV/VH分别处理。

每张查询在冻结的64个**源端**参考中按候选类各检索2个案例，不需要知道查询类别。64=8类×每类8个源参考，**不是需要用户再提供64个目标标签**，也不是查询缓存。

`m3.py` 的 `PairMetric` 根据查询—参考联合描述调整32维内容比较权重；同一相似度同时进入带未匹配槽的 Sinkhorn 计算，故 both 版既改变“怎么比”，也影响“对应到哪里”。船体、侧向、轴端分别匹配，不随意跨区域混配。

匹配摘要经过独立 VV 和中心化 VH 读出。185维门控输入包括11维分数信息、30维空间关系及144维频域关系，围绕冻结锚点作有界调整：`alpha = clip(alpha_B + 0.25*tanh(r), 0, 1)`。双极化缺失时保留原锚点逻辑并屏蔽无证据极化；最终不是固定50/50，也不强行让单极化权重归一到1，因为那会改变已选模型。

## 4. 输出与可追溯性

默认输出类别编号、类别名、8类分数、softmax分数、VV/VH权重、所用检查点步数、输入裁剪/补零记录及目标证据可用性。**softmax不是已校准置信概率**，输出明确标记这一点。

`--trace` 额外输出基础分数、M1/M2增量、所选源参考ID、局部匹配摘要、VV/VH增量和门控输入。可据此追踪判别链，但这些数值本身不等于物理因果解释。

类别顺序：散货船、渔船、普通货船、成品油/化学品船、集装箱船、原油轮、拖带船、海工供应船。当前权重固定八分类，不含拒识类，不能直接作为 F_0921 新细分类权重或非船过滤器。

## 5. 代码位置

| 文件 | 职责 |
|---|---|
| `sar_m123/input.py` | 图片、检测框、原生裁补和极化可用性 |
| `sar_m123/backbone.py` | 共享冻结R50及C4多层特征 |
| `sar_m123/baseline.py` | 实际旧基础路径 |
| `sar_m123/m12.py` | M1、M2响应读出与分类增量 |
| `sar_m123/regions.py` | OBB软区域、质量与有效性 |
| `sar_m123/tokens.py` | M3局部槽位与描述构造 |
| `sar_m123/frequency.py` | Gabor及联合频域描述 |
| `sar_m123/m3.py` | 查询—参考比较、匹配及极化融合 |
| `sar_m123/pipeline.py` | `ShipClassifier`公开接口 |
| `sar_m123/cli.py` | JSON命令行输入输出 |
| `tools/export_legacy.py` | 一次性导出已有权重，不是推理依赖 |
| `tools/validate_raw.py` | 原chip到分类与历史结果对照 |
| `tools/check_portability.py` | 移目录并禁读旧工作区的CPU验证 |

资源包含骨干、冻结源端读出器、标准化参数、PCA投影、参考特征及三个配置检查点。源参考库是该模型推理所需资源，不是可随意删除的训练日志。

若本机资源丢失，才需要从历史目录重新导出：

```powershell
python tools/export_legacy.py --legacy-root "E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023" --backbone "C:/Users/25791/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth"
```

历史迁移工具额外使用 pandas；日常推理不需要。只克隆Git仓库不足以得到权重，不能在没有 `assets/` 的情况下直接推理。

## 6. 验证结果与研究边界

详见 `validation_reports/ENGINEERING_VALIDATION_ZH.md`。8张真实原图、3个配置共24次端到端预测与历史结果全部相同，最大分数差0.00011659；独立目录CPU禁读旧工作区测试也通过。不是只用预存特征重放。

保留原选择事实：配置17的M1选中0步、M2选中15步；29为70/0；43为10/0。工程包含全部模块，但**不能声称三个配置里三个模块都带来独立叠加收益**。没有为让模块“看起来都起作用”而改权重。

本轮完成的是独立的、已训练模型的完整推理链，不是把全部历史研究代码复制过来：**不包含从头训练/重新选择检查点、检测器训练与整景滑窗检测、原始SAR辐射处理、自动港口地理关系恢复**。检测器冻结后产生chip和OBB，即可接入本工程；不再需要查询特征缓存。当前未新增训练、未扩大科学实验覆盖，也不因工程化就提升因果或创新性主张。
