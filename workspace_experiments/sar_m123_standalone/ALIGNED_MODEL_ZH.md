# 新版独立模型工程

## 1. 冻结的是哪个版本

本版对应 2026-10-26 已接受的 `raw_summary + adapted PairMetric + paired readout refit`，输出端为 `frequency`。保留 OSS1 源模型、OSS2 源模型各三个源端选定配置（17、29、43），四类顺序为散货、普通货、化学品、集装箱。

它不是旧正式八分类模型的重命名，也没有保留旧 M1→M2 逐级加分的全部结构。旧 `ShipClassifier` 与 `assets/` 保留；新版使用 `AlignedClassifier` 与 `assets_aligned/{oss1,oss2}/`。当前工程没有把公开四分类的训练结果自动扩成八分类。

## 2. 图像到输出

```text
VV / VH chip + 各自目标 OBB
  -> 原生像素裁剪/补零到160，不缩放船体
  -> 冻结R50，四个90度旋转分别前向
  -> VV / VH 分开处理
     ├─ M1: layer1类别响应，均值及双向注意力摘要
     └─ M2: OBB组织框内、侧向、轴端区域，保留区域身份
  -> 原始区域特征1538维 + 注意力/质量/几何摘要104维
  -> 源端标准化 -> 128维MLP -> 四类基础分数
  -> 每个候选类别从源参考库检索两个案例
  -> M3: 学习比较度量 + 可微对应匹配
  -> VV校正、空证据中心化VH校正
  -> 空间—频域条件化极化权重
  -> 基础分数 + VV校正×权重 + VH校正×权重 -> 分类
```

注意两种区域顺序不同：原始1538维及M3局部槽位是框内/侧向/轴端；M1摘要是框内/轴端/侧向。代码分别保持原实验顺序，没有直接混拼成同一槽位。

104维由72维类别响应摘要、24维区域质量与8维几何字段组成。不能将全部104维带来的收益都归功于注意力。几何字段不是经过验证的真实船首/船尾标签。

`frequency` 是输出端名称，不能据此断言动态频域选择已起作用：现有六个保留配置的融合门均选中零步初始化，与各自的固定B比例一致。比较度量仍使用联合描述，动态门的贡献须与它分开讨论。

## 3. 核心代码

| 文件 | 内容 |
|---|---|
| `sar_m123/input.py` | 原生chip、OBB、缺失极化、裁剪补零 |
| `sar_m123/backbone.py` | 冻结R50与C4特征 |
| `sar_m123/aligned.py` | 新版M1/M2读出、基础分类、四分类M3及完整API |
| `sar_m123/m12.py` | 复用类别响应注意力基础算子；旧完整链另行保留 |
| `sar_m123/regions.py`、`tokens.py` | OBB区域、局部槽位、面积与响应读出 |
| `sar_m123/frequency.py` | 空间—频率描述计算 |
| `sar_m123/m3.py` | 复用比较度量、软匹配、极化分支基础算子 |
| `sar_m123/prepare_training.py` | 原图清单生成可移植训练特征 |
| `sar_m123/train_aligned.py` | 新版分阶段训练入口 |
| `sar_m123/train_ops.py` | 分类损失、源端选模、可微匹配和极化训练 |
| `sar_m123/training_alignment.py` | 沿实际决策路径训练比较度量 |
| `tools/export_aligned.py` | 一次性导出历史权重，或安装本工程重训结果 |

运行模型、提取新chip特征与训练时不导入历史实验目录。`tools/import_training_bundle.py` 和资源导出器仅用于一次性迁移已有文件，不是推理依赖。权重、原图和参考特征不进入Git。

## 4. 推理

安装现有工程：

```powershell
python -m pip install -e ".[training]"
python -m sar_m123 --model aligned --assets assets_aligned/oss1 --input request.json --output outputs/aligned.json --seeds 17 29 43 --trace
```

输入格式沿用 `examples/request_template.json`，图像路径相对输入JSON。提供VV/VH图像与检测OBB；无有效目标框的极化不伪造局部证据。仅有VV也可输入，但分类不自动重标定为已验证的单极化模型。

三个seed表示三个已训练配置，不是运行时重新训练，也不是自动集成。输出含各配置类别分数、预测类别、极化权重；`--trace` 另含基础分数、两支路校正、匹配关系及实际参考索引。softmax输出没有自动成为校准概率。

Python API：

```python
from sar_m123.aligned import AlignedClassifier

model = AlignedClassifier("assets_aligned/oss1", device="cuda")
result = model.predict(record, directory=chip_directory, seeds=(17,), trace=True)
```

部署不需要查询标签、AIS类别或港口真值。`mmsi` 仅在源样本重放时用于排除同船参考；在真正不重叠的外部查询可不提供。工程未包含整景检测、开放集拒识或地理设施图模块。

## 5. 训练与再导出

这是**冻结源端预训练前端的分阶段训练**，不是从零预训练R50或重新构建所有上游资产。类别响应读出器、PCA、32个源参考及初始PairMetric是明确的预训练输入；它们必须来自合法源数据。当前发布实现与验证针对已有四分类配置，不声称新数据源上所有上游预训练已一键重现。

训练清单是JSON列表。每项除推理所需图像与OBB外，增加：`id`、`mmsi`、`pass_group`、`role`、`class_id`。role为 `fit_train`、`fit_validation`、`utility_train`、`selection` 或 `evaluation`。四个训练/选模角色各覆盖四类，船只及采集组隔离；evaluation标签不进入训练。清单须包含该源模型的32个冻结参考观测。

```powershell
python -m sar_m123.prepare_training --input source_records.json --assets assets_aligned/oss1 --output training_bundles/oss1 --device cuda
python -m sar_m123.train_aligned --bundle training_bundles/oss1 --assets assets_aligned/oss1 --output training_outputs/oss1 --seeds 17 29 43 --device cuda
```

训练步骤固定，不依据测试结果搜索：

1. 在fit_train上拟合两组标准化参数及raw-summary分类头，200步；fit_validation选择检查点。
2. 冻结初始PairMetric，拟合VV、中心化VH与融合门，训练/选模分别用utility_train/selection。
3. 通过实际冻结极化决策训练PairMetric，最多100步；不把中间相似度误差当作最终分类目标。
4. 用选中匹配器重新拟合极化读出和融合门，保存最终权重、分数与选模记录。

损失和选模沿用原实验：基础头按类别与船只重复数加权；M3使用OA/BA混合权重，选择准则是精确计算的 `(OA+BA)/2`，只接受严格改善，保留零步候选。

将训练结果安装到**资源副本**，保留原发布资源以便对照：

```powershell
python tools/export_aligned.py --assets-root trained_assets --source oss1 --training-root training_outputs/oss1
```

`trained_assets/oss1` 须先具有原资源的前端、参考库和warm初始化；该命令更新分类头与最终M3。迁移到另一台机器时，需要代码、依赖和相应资源目录，不需要历史实验文件夹。

一次性迁移已有本机成果的命令：

```powershell
python tools/export_aligned.py --legacy-root MODEL_CLOSEOUT_ROOT --assets-root assets_aligned
python tools/import_training_bundle.py --legacy-root MODEL_CLOSEOUT_ROOT --source oss1 --output training_bundles/oss1
```

GitHub只含代码、说明和小型验证报告。预训练权重/源参考特征需要另外取得或在本地导出；仅克隆仓库不会自动获得它们。

## 6. 验证口径

打包验证与研究验证分开：本轮核对缓存输入重放、原生chip前向、分阶段训练及部署结果，不以它们作为新的泛化收益实验。数值允许明确报告的浮点误差，但参考对应、选中步骤和预测变化必须核查；未验证项不得宣称完成。

最新研究结论仍以数据集内跨采集、跨数据集配对消融报告为准。当前版本有范围内的收益，但没有全面超过所有简单对照。原始chip解释案例复用已审阅的样本，不因工程迁移重新命名物理机制。
