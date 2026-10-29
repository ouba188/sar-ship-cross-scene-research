# OSS2 最终空间 M12 vs plainR50 四类复现

日期：2026-10-29。协议在结果产生前固定。六条正式训练链、评价、统计和原图回看已完成；原支线准备记录见CHECKREADINESS_ZH.md，最终结果见FINDINGS_ZH.md及训练收据。

## 固定协议

- 使用 `../oss2_m123_replication_20261026/` 的历史 fold0 检测后 stack。只转换现有 CSV；不重检、不下载、不沿用旧 C4 特征或旧训练权重。
- 明确四类顺序：`bulk_carrier/general_cargo/product_chemical_tanker/container_ship`。原八类输出行切片为 `[0,2,3,4]`，不是八类复现。
- 初始化唯一来源：`E:/临时会话/sar_m123_current_20261027/weights/init.pt`。复用相邻 `final_module_ablations_20261029/model.py` 的 `fresh/save/load/fitting_loss`；不导入其 run。17/29/43 是共享初始化下的 minibatch 顺序，不是独立初始化。
- full 保留最终空间 M1/M2 与原 M3；plain 为共享 VV/VH R50、有效像素 GAP、两路 available 标志和四类线性 head，无参考值作用。
- sample_id 为持久身份；显式保留 obs_id、manifest_position、target_manifest_row、legacy_row_index、table_index、row_index、up_index。reference_bank 按 obs_id 连接回 stack，再用 sample_id 定位缓存。
- OBB corners 使用 raw_xywhr 的原始 width/height/angle；排序后的长短轴只作为已有 ship_axis。无有效目标的极化为 null；至少保留一条有效极化。雷达角一律 null，不从 heading 推断。
- 原生 160 crop/pad，无 resize、C4 或 224 输入。保留大 chip 的既有 crop 行为；原图为 uint8 显示 chip，不声称标定散射数据。

| 用途 | 角色 | 观测 | 双路/仅VV/仅VH |
|---|---|---:|---|
| fit | fit_train | 478 | 444/11/23 |
| adapt | utility_train | 241 | 227/5/9 |
| 仅记录可用 | fit_validation | 228 | 206/6/16 |
| 仅记录可用 | selection | 194 | 177/5/12 |
| 主评价 | evaluation | 378 | 348/12/18 |

参考固定 32 个独立 MMSI，每类 8 条，均来自 fit。每次调用仍按 id/MMSI 排除同船参考，标签仅用于合法训练损失。两个开发角色保留清单与数量，不输出开发分数、不挑 checkpoint。

## 训练与共享边界

两臂均 fit 8 epoch、batch24、160 steps；adapt 6 epoch、batch16、96 steps。AdamW decay=.01；fit 的模块/head lr=.001、三个 LIVE 卷积 lr=3e-5；adapt 仅三个 LIVE 卷积 lr=3e-5。复用 current.training_policy，BN/anchor/M3 readers 的冻结规则一致。

fit 使用每船重复观测补偿，分母明确为 4，按 batch mean 归约；adapt 使用四类平衡权重，按权重 sum 归约。full 的 fit 为 CE+.1aux+.1relation，plain 仅 CE，空 aux/relation 为同设备标量零。

每条 full 链用自己的 fit 端点调用原 `current.interventions.select_records`，保持原干预、mean control、E1/K0/F0 复核；adapt 系数 `.3683066666126251`。plain 用自己的原教师，useful=False、coefficient=0，仅 CE+.1正确教师KL。保存固定 fit/adapt 两端点。该比较包含架构与配套训练目标的差异，不声称单独隔离架构效果。

`run.py cache` 按 8 条一批生成唯一 float32 layer2 prefix 及原生 masks/valid/available/slots/frequency/field2/field3。缓存形状 prefix=`[1519,2,512,20,20]`；包含所有固定角色，存储纯输入信息不含拟合标签特征。文件采用 memmap，读取只复制请求 batch，参考一次载入32条；不将全量数组装进 RAM 或 GPU。不缓存 LIVE/M1/M2/M3 输出、live reference values 或 logits。full 原教师选择必须重算被干预图像的前缀，直接沿用原 API；该必要重算不等于重提旧 C4。

## 使用方法

所有命令在本目录执行；以下前三项已经执行过。使用 `-B`，共享模块导入也禁用字节码落盘，避免在外部目录生成 pycache。

```powershell
python -B -m unittest tests_cpu -v
python -B converter.py
python -B checkreadiness.py
```

主代理审查 `tests_gpu.py` 后已执行以下流程；最初准备支线没有使用GPU：

```powershell
python -B run.py cache
python -B run.py check
python -B run.py all
```

`all` 会复用已有完整缓存，再实际运行 GPU check，然后串行执行 full/plain × 17/29/43 的 fit→select→adapt→evaluate。每条链独立固定端点；遇到已有模型/教师文件会停止而不覆盖。分阶段继续可用 `run.py fit|select|adapt|evaluate --arms full --seeds 17` 等命令。若之前已完成部分链，显式选择剩余臂/seed/阶段，不直接重跑 all。

check 的必要条件包括：4类输出、head与两路M1 reader源行切片、plain全局head列、有效极化与原生160、rawangle测试、所有固定输入字段/native缓存对照、参考排除、plain无参考作用、LIVE和存活模块梯度、空损失/无合法参考、margin分母3、plain CE+KL值与梯度、严格保存恢复。GPU checks 成功才写 status=passed；异常保留 running，不生成成功收据。训练固定端点再次检查冻结状态、LIVE更新、保存重放、native/cache一致。

## 浮点检查修正

首次GPU检查在内部prefix的绝对误差阈值1e-4处中止，尚未训练。诊断见 `cache_roundoff_diagnosis.json`：按生成缓存时的原8条batch重放，7个诊断查询误差均为0；改变batch为单条时内部特征最大差约0.00014至0.00018，而特征幅值约142至165。非prefix输入字段完全相同。原图查询和原图参考共同前向的分类分数差4.4107e-6，top-1不变。

修正后对prefix增加原batch重放检查（<1e-5），另记录跨batch浮点差；最终分类分数阈值仍为1e-4、保存恢复仍为1e-5，没有放宽这两个输出检查。完整GPU检查通过后才开始训练。原失败run.log、诊断脚本和run_retry.log保留；未改变训练超参数或挑选结果。

## 输出和后续边界

- CPU 已产出：`manifests/*.json`、`conversion.json`、`cpu_readiness.json`、本协议及检查状态说明。
- GPU 已产出：`cache/*.npy`、`cache/complete.json`、`checks/*.pt`、`gpu_checks.json`。
- 训练已产出：`training/{arm}_s{seed}/{fit,adapt}/model.pt` 与 training.json；`teachers/*.json`；`predictions/evaluation_{arm}_{fit,adapt}_s{seed}.npz`；`evaluation/*.json`。
- npz 使用 `score/ids/y/classes`。同行共享 sample_id 与固定四类顺序，便于主代理配对 OA/BA/NLL/Brier、逐类和过境分析。
- 三顺序配对结果、真实同类/混淆参考及16面板回看已完成；正负结果均保留。原始数据、权重、NPZ及完整清单不上传GitHub，代码、统计、收据和派生案例归档。
