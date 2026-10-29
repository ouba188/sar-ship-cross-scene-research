# OSS2 完整模型接入说明

2026-10-30。只读核查现有实现、清单和训练收据；仅新增本文。未修改现有代码、训练或使用 GPU，未重复实现 M3 修复。已阅读根目录 AGENTS.md、codex-brain/INDEX.md 及 04-workflows.md 的模块组合规则。本支线不操作 Git，由主代理统一归档。

## 1. 模型与 checkpoint

下文 `ROOT` 为 `E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/final_oss2_replication_20261029`。

- 完整最终模型：`ROOT/training/full_s{17,29,43}/adapt/model.pt`；各自适配前端点为同目录的 `fit/model.pt`。六个文件均存在，每个约 112 MB。保护旧回写的最终模型对照应从匹配顺序的 **adapt** 端点开始；fit 仅用于明确的适配前对照。
- 正确加载器：`E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/final_module_ablations_20261029/model.py` 中的 `load(path, device)`。其 `AblationModel` 接受 `arm='full'`，按 checkpoint 的 `classes` 建模并严格加载 `state`，返回 eval、全部参数冻结的模型。
- checkpoint 架构为 `native160_spatial_m12_train_time_ablation_v1`，四类标签 0/1/2/3 对应 `bulk_carrier/general_cargo/product_chemical_tanker/container_ship`，原八类行号为 `[0,2,3,4]`。不要再次切片最终权重，也不要使用八类 `current.model.load_model()`，后者会拒绝此架构。
- 初始化来源是 `E:/临时会话/sar_m123_current_20261027/weights/init.pt`；只在重建 fresh 模型时需要它，加载完整端点无需先加载 init。17/29/43 是共享初始化的 minibatch 顺序，不是独立初始化。
- full 的 fit 固定 8 epoch/160 steps，adapt 固定 6 epoch/96 steps；六份收据均 `validation_selection=false`。本次不重新选择原模型端点。

## 2. 可用于源端筛选的数据

直接读取 `ROOT/manifests/<文件>`。源端计数和类别数已从现有 JSON 核对；隔离依据为现有 `cpu_readiness.json`、`checkreadiness.py` 和上游 `STACK_SPLIT.json`。

| 文件 | 观测/船/过境 | 0/1/2/3 类观测数 | 角色与建议用途 |
|---|---:|---|---|
| source_fit.json | 478/448/9 | 138/47/95/198 | 原 fit_train，bag2/3；修复训练候选，已被基础模型训练使用 |
| source_adapt.json | 241/235/4 | 83/26/46/86 | 原 utility_train，bag5；已参与原模型适配，不能作为独立验证 |
| validation.json | 228/223/4 | 87/23/29/89 | 原 fit_validation，训练侧 bag4；可用于 M3 候选、停止点或保护强度筛选 |
| selection.json | 194/193/5 | 64/18/40/72 | 历史 fold0 验证包 bag1；可保留作候选冻结后的源端复核 |
| references.json | 32/32/9 | 8/8/8/8 | 固定 source_fit 参考库，保持现有身份与顺序 |

这五个实验角色中的 source_fit/source_adapt/validation/selection/evaluation 之间，现有检查确认 id、MMSI、pass_group 两两无交集；references 是 source_fit 子集，不是额外独立证据。`validation` 和 `selection` 在 **2026-10-29 这一轮完整模型实验** 中仅登记可用，未输出开发分数或挑 checkpoint；旧实验存在历史开发暴露，不能称为从未使用的盲测数据。

建议本轮明确固定：训练仅使用约定源训练集，在 validation228 比较候选并保留“不修改”选项，冻结后用 selection194 复核；若根据后者继续改方案，它也成为开发集。最终 evaluation378 为历史测试包 bag0，只能在方案冻结后评价，不参与阈值、类别策略、参考选择或挑顺序。上述是本轮可采用的分工，不是声称旧 runner 已执行过这些筛选。

注意同名概念：`run.select_one()` 实际在 **source_adapt241** 上调用 `current.interventions.select_records()` 生成干预教师，与 `selection194` 留出集无关。`teachers/full_s*.json` 是各自 fit 端点的源端干预证据，不是最终 adapt 模型的 M3 修复标签。`select_records()` 内部 `assert_source()` 只接受 source_fit/source_adapt；不要修改 validation/selection 的 split 来绕过它。留出集筛选应直接用预测和其源端标签计算指标。

## 3. 不读取测试标签的最小调用

复用上述 `load()`、`current.data.read_manifest/pair_mask` 和模型 `forward_from()`。`run.Cache.batch/score` 的缓存机制可以沿用，但其构造器读取 `all.json` 与含全体标签的 `cache/complete.json`；`protocol.cohorts()` 也会读取 evaluation 清单。严格只读源端时使用下面的源清单索引方式，不调用这两个入口。

```python
import sys
import importlib.util
from pathlib import Path
import numpy as np
import torch

sys.dont_write_bytecode = True
ROOT = Path("E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/final_oss2_replication_20261029")
sys.path.insert(0, "E:/临时会话/sar_m123_standalone")
from sar_m123.current import data, model

device = "cpu"
model.setup(device)
spec = importlib.util.spec_from_file_location(
    "oss2_handoff_model", ROOT.parent / "final_module_ablations_20261029/model.py")
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)
net = api.load(ROOT / "training/full_s17/adapt/model.pt", device)
queries = data.read_manifest(ROOT / "manifests/validation.json")
refs = data.read_manifest(ROOT / "manifests/references.json")
keys = (set(data.SHAPES) - {"images"}) | {"prefix"}
arrays = {k: np.load(ROOT / "cache" / f"{k}.npy", mmap_mode="r") for k in keys}

def take(records):
    ix = [r["indices"]["table_index"] for r in records]
    return {k: torch.as_tensor(np.array(a[ix], copy=True), device=device)
            for k, a in arrays.items()}

bank = take(refs)
group = queries[:1]  # Iterate bounded batches for the chosen source split.
with torch.no_grad():
    score, detail = net.forward_from(
        **take(group), reference=bank,
        pair_valid=data.pair_mask(group, refs, device), return_details=True)
```

这里仅加载源清单，缓存数组本身没有标签。不能用 `obs_id`、`manifest_position` 或 `target_manifest_row` 替代 `table_index`：例如首个 validation 样本 `pub2_opensarship2_000352` 的缓存行号为 478，而 obs_id 为 160。固定缓存中 source_fit 为 0..477、validation 为 478..705、source_adapt 为 706..946、selection 为 947..1140；调用仍按记录字段取行，不依赖硬编码范围。预测输出为 logits `[B,4]`，不是概率；配对用 `id`。

## 4. 原始 chip、缓存与 M3 接口

- 原始模型输入路径在 `record['vv'/'vh']['image']`，根目录为 `D:/Documents/project_root/project_data/public_opensarship2/opensarship2/chips/{vv,vh}/`。`source_paths` 是更上游 TIFF 来源记录，不是当前 PNG 输入入口；极化为 null 时不能用上游图像补成有效极化。
- 原图路径：`item, views = data.prepare_record(record, return_views=True)`，其中 `views[p]['raw']` 为原始 uint8 显示 chip，`pixels` 为 native160 crop/pad，`placement` 记录裁切与补边。VV/VH 顺序固定，分开查看；输入除以 255，不 resize、不 C4、不换 224。雷达角为 null，OBB/ship_axis 已提供，无需重检。这里的“原图”不是标定 SAR 散射数据。
- 无缓存推理可直接调用 `model.predict_records(net, queries, refs, device='cpu', batch_size=...)`；需要详情则用 `reference_bank()`、`data.batch()`、`forward_batch(..., details=True)`。这些函数可以接受已正确加载的四类模型，不需调用通用八类加载器。
- `cache/prefix.npy` 是 float32 `[1519,2,512,20,20]` 的冻结 layer2 输出，文件约 2.49 GB，应用 `mmap_mode='r'` 按批取。其他数组提供 masks/valid/available、slots、frequency、field2/3。M3 修复若只改 layer2 后的模块，可复用此缓存；改变图像或 layer2 及以前的权重时不能直接复用对应 prefix。
- 缓存没有 M1/M2/M3 的已学习输出；参考输入固定，但实际 reference values 在每次前向中随当前模型重新计算。不要把旧端点的 M3 value 输出当作新候选的缓存。空间 M1/M2 不使用频谱分支，但现有 M3 仍使用频率证据，不能删除这些输入字段。
- 原模型只有一组 M3，VV/VH 为 `net.m3[0][0/1]`。`detail['operators']['m3_0'][p]` 包含 `update`、`reference_logits`、`reference_valid`、`reference_weights`；当前实现返回 `x + update`。主代理可据此接入保护旧回写的对照，不需改公开 runner。`pair_mask()` 排除同 id/MMSI，模型再叠加查询/参考的极化可用性；保留两层逻辑。
- `load()` 默认全冻结；旧 `training_policy(net, 'adapt')` 只解冻三个 LIVE 卷积，不会训练 M3。新修复参数策略应由主代理明确设置，不能直接套用旧 adapt 策略。新增模块参数的修复 checkpoint 也需相应的修复加载器，原 `api.load()` 只支持原结构。

## 5. 核验结果与限制

实际只读 CPU 核验使用 `D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe -B -X utf8`：成功加载 full_s17/adapt，确认 full/4 类/一组双极化 M3/参数全冻结；读取上述 validation 样本的 VH 原图为 49x49、VV 不可用；源端单条查询加实际 32 参考的缓存前向成功，输出 `[1,4]` 且有限，M3 update 为 `[1,1024,10,10]`，`torch.cuda.is_initialized()` 为 False。没有运行测试集预测、候选筛选或训练，也未评估修复收益。

其余五个 full checkpoint 核对了文件存在与对应收据，没有逐一重跑。现有六份保存重放误差均低于 1e-5、native/cache logits 误差均低于 1e-4；不要求跨 batch 内部 prefix 位级一致，既有浮点差说明见 `ROOT/cache_roundoff_diagnosis.json`。

本机默认 `D:/Program_files/anaconda3/python.exe` 在尝试只读 torch 加载时触发 OpenMP Error #15；项目指定 ClearSAR 环境通过，无需安装依赖或设置重复 OpenMP 绕过变量。不要运行旧 `run.py all/fit/select/adapt` 完成接入：这些入口会启动原实验或写入已有目录，并不实现本轮 M3 对照。
