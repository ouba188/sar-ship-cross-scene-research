# PACS / Office-Home 模块补齐记录

日期：2026-10-08。

## 当前执行状态：Office-Home 四域全部完成

主代理六港 GPU 窗口结束后，本支线按 Art、Clipart、Product、Real World 顺序串行完成 bootstrap → prepare → train → evaluate。复用了原图与 uint8 Resize224 缓存，没有重新提取图像。每域固定 batch64、bootstrap12轮、M3/joint各6轮；协议 `retrained_rgb_reference` 单列，原 Base64 与五组共享64标签支持集保持不变，源参考和门控在 M3 前冻结。无论结果正负均保存，未根据目标结果调参。

11:29:44 GPU 训练与提取全部结束，已写 `GPU_RELEASED.txt` 并通知主代理接手剩余 SAR 港口；11:30:14 最后 CPU evaluate 完成。控制进程 PID=73988 正常退出，已确认无本支线 `generic_joint_completion.py` 进程。四域各四臂五 draw，共80条；训练 history 均核对为12/6/6轮。主代理 `CORE_MATCHED_RESULTS`、`module_coverage_completed` 未修改。

### Office-Home 四域完整读出

下表每格为 Full OA / BA（%，同一域五个 draw 的均值）：

| 外层目标域 | 新 reference | M2 complement_only | M3 aligned_only | joint |
| --- | ---: | ---: | ---: | ---: |
| Art | 66.271689 / 62.761934 | 66.220906 / 62.760844 | 65.433771 / 62.137346 | 65.103682 / 61.731630 |
| Clipart | 51.141595 / 51.091555 | 51.122995 / 51.089321 | 50.625436 / 50.572495 | 50.378982 / 50.365397 |
| Product | 74.742857 / 72.891382 | 74.756571 / 72.893404 | 74.066286 / 72.220950 | 73.842286 / 72.018047 |
| Real World | 76.356860 / 74.825484 | 76.286979 / 74.748208 | 74.884696 / 73.299787 | 74.567901 / 72.946789 |

先在每域内对五 draw 取均值，再对四域等权平均，不按查询样本数加权：

| 方法 | Full OA % | Full BA % | 对新参考 OA pp | 对新参考 BA pp | Expert OA % | Expert BA % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 新 reference | 67.128250 | 65.392589 | 0 | 0 | 64.835052 | 63.428996 |
| M2 | 67.096863 | 65.372944 | -0.031388 | -0.019645 | 64.792148 | 63.372163 |
| M3 | 66.252547 | 64.557644 | -0.875703 | -0.834945 | 55.675122 | 54.240560 |
| joint | 65.973213 | 64.265466 | -1.155038 | -1.127123 | 54.897486 | 53.497736 |

不交接的原 Base64 等域 OA/BA=66.558133%/64.852800%，四臂完全相同。这里 Expert 指对全部查询的专家预测，不是仅交接子集。M3 与 joint 的完整系统 BA 在四域全部下降，每个域的五 draw 也全部下降。联合专家 BA 均值下降9.931260 pp，完整交接系统 BA 下降1.127123 pp。因此不能将当前冻结协议的联合模块写成普通 RGB 跨域任务上的普遍增益。M2 仅 Product 均值略正，总体基本持平略负；旧参考 M2 的微小正收益与本次新参考略负分开报告，不互相替代。

机器结果：`rgb_retrained/officehome/summary.json`（`dataset_complete=true`）、同目录 `metrics.json` / `metrics.csv`。四个域子目录分别保存 RGB_PROTOCOL、bootstrap/双臂训练日志、权重、prefix、冻结源模型与门控、episode、四臂策略以及全部逐 draw 指标。

| 外层域 | bootstrap12轮 秒 | M3六轮 秒 | joint六轮 秒 |
| --- | ---: | ---: | ---: |
| Art | 223.859 | 24.609 | 25.235 |
| Clipart | 199.000 | 18.594 | 18.812 |
| Product | 176.735 | 20.562 | 22.141 |
| Real World | 179.391 | 21.891 | 26.468 |

日志训练时间共957.297秒（约15.955分钟）；串行全流程墙钟从11:07:09.660至11:30:14.182，约23分04.522秒。后者包括环境导入、CPU prepare、prefix/特征提取和评估；二者均不含此前一次性缓存准备时间。未修改核心训练代码，未启动额外 GPU 实验，未遗留本支线待完成的 PACS/Office-Home 实验。

## 已完成：M2 匹配实验

两数据集全部四个外层目标域，每域五个原存档支持集；每次同一组 64 个目标标签，5% 查询交接上限。CPU threads=2，按域顺序执行，无 GPU 训练。

每个目标域保持原基础分支、源分类头、源 OOF 门控参数不变，仅替换专家残差核：reference = K_retained + 1，complement_only = 0.5 K_retained + 0.5 K_discarded + 1。补空间标准化参数仅由该外层源域计算。全部 80 条策略已保存，随后读取查询标签计算指标。

参考版与旧存档的最终预测及交接集合全部精确一致；专家得分最大数值差小于 1.1e-7。结果仍是条件于现有 RGB 编码器的机制验证，不是 SAR 编码器迁移或独立盲测。

| 数据集 | reference OA | M2 OA | OA 差值 pp | reference BA | M2 BA | BA 差值 pp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PACS | 77.969497 | 77.992571 | +0.023073 | 79.868998 | 79.882573 | +0.013575 |
| Office-Home | 67.285482 | 67.310114 | +0.024632 | 65.530986 | 65.564288 | +0.033302 |

均为四域等权平均，不按样本数加权。PACS 仅 2/4 域 BA 改善，Office-Home 4/4 域改善。两个数据集的全局专家均值没有同步改善：PACS 专家 OA/BA 分别变化 -0.136547/-0.160360 pp；Office-Home 为 -0.032918/-0.001594 pp。因此不能称 M2 在普通 RGB 域上带来明显判别信息增益；目前只是完整交接系统的微小平均增益。

机器结果：本目录 `summary.json`、`metrics.json`、`metrics.csv`。逐域子目录保存原协议来源、源几何和两臂全部策略。指标 JSON 中准确率为 0~1；`delta_vs_reference_pp` 为百分点。

## 已完成：首个新 RGB 四臂匹配实验

PACS / art_painting，独立协议 `retrained_rgb_reference`；源域为 cartoon/photo/sketch，完整源集7943行。ImageNet-V1 R18 重新监督训练12 epoch；目标域标签未用于编码器、源头或门控训练。原 Base64 得分保持不变。5组64标签支持集，1984个查询/组，最多99个专家交接/组。

| 方法 | Full OA % | Full BA % | 对新参考 OA pp | 对新参考 BA pp | Expert OA % | Expert BA % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 新匹配 reference | 81.239919 | 81.134362 | 0 | 0 | 75.796371 | 76.953483 |
| complement_only / M2 | 81.350806 | 81.260893 | +0.110887 | +0.126532 | 75.625000 | 76.794812 |
| aligned_only / M3 | 80.141129 | 80.127457 | -1.098790 | -1.006905 | 67.026210 | 68.682600 |
| joint / M2+M3 | 80.040323 | 79.977943 | -1.199597 | -1.156419 | 66.774194 | 68.402495 |

这一域 M2 略有收益，M3 与联合明显下降，不能写成联合模块跨 RGB 域普遍有效。源侧训练损失下降不能替代目标效果；本结果不触发针对 art_painting 的超参数、训练轮次、域选择或门控调参。其余三个 PACS 域现已按同一冻结协议完成，四域均值见下一节；Office-Home 四域完整结果见本文开头。

阶段训练计时：bootstrap 十二轮177.36秒，aligned-only 六轮20.19秒，joint 六轮19.38秒；不含各阶段环境导入、CPU几何、prefix/最终特征提取和查询评估。M3 使用1.003GB原值float16 GPU prefix缓存；所有输入大小和训练数据/轮数不变。两数据集 Resize224 缓存均完成，PACS51.3秒、Office-Home532.0秒，缓存成本只支付一次，供四个外层域共享。

保存于 `rgb_retrained/pacs/art_painting/`：协议、训练日志、RGB源权重、prefix、新源模型与门控、源episode、两个layer4权重、四臂完整策略、逐draw指标及summary。其余三个 PACS 域在各自同名目录保存同样的完整产物。

## 已完成：PACS 四域联合模块汇总

art_painting、cartoon、photo、sketch 全部完成 bootstrap 12 epoch、M3 与 joint 各 6 epoch，以及四臂各 5 draw 查询评估，共 80 条指标。协议为 `retrained_rgb_reference`，不是旧数值的精确重放。每域沿用原 64 支持标签和 Base64，外层目标域排除于源训练，源参考与门控在 M3 前拟合冻结；没有根据目标结果调整超参数。

| 方法 | 四域等权 Full OA % | 四域等权 Full BA % | 对新参考 OA pp | 对新参考 BA pp |
| --- | ---: | ---: | ---: | ---: |
| 新匹配 reference | 77.809164 | 79.635512 | 0 | 0 |
| complement_only / M2 | 77.813689 | 79.636341 | +0.004524 | +0.000829 |
| aligned_only / M3 | 77.349439 | 79.266995 | -0.459725 | -0.368517 |
| joint / M2+M3 | 77.280367 | 79.184320 | -0.528798 | -0.451192 |

联合版在四域 BA 均下降，M2 平均效果基本持平。不能据此声称联合模块普遍提升 RGB 跨域识别。所有负结果保留，不删域、不按目标结果选轮次或重新拟合门控。

机器汇总为 `rgb_retrained/pacs/summary.json`，`dataset_complete=true`；完整逐域逐 draw 记录为该目录 `metrics.json` 与 `metrics.csv`。逐域结果另见 `PACS_STATUS.md`。四域日志内训练时间合计 744.720 秒（约 12.41 分钟），不含环境导入、CPU prepare、prefix/特征提取与 evaluate，不能作为全流程墙钟耗时。

PACS 完成后曾释放 GPU 给主代理六港训练；六港窗口结束后，Office-Home 四域也已全部完成。最新释放时间及结果见本文开头与 `GPU_RELEASED.txt`。

## 后续 RGB 库存

已确认原图在本机：

- PACS：`E:/Hermes/xdomain_data/pacs/train-00000-of-00001.parquet`，9991 行，191395900 字节。
- Office-Home：`E:/Hermes/xdomain_data/office_home/` 下三个 parquet，15588 行，依次 248782886 / 119438711 / 790762518 字节。
- ImageNet-V1 RGB R18 初始化：`C:/Users/25791/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth`，46830571 字节。

旧外层域监督 R18 checkpoint **未保存**：`xdom_pacs/run_pacs.py` 和 `xdom_generic/run_xdom_generic.py` 仅保存最终 512D 特征，没有 `torch.save`；对应项目目录不存在 `.pt/.pth/.ckpt` 或 layer3/layer4 缓存。进一步按 PACS/Office 命名搜索 Hermes 目录也未发现对应文件。不能由已池化的 512D 特征恢复原 layer4/prefix。

因此 M3/联合版不能直接接续旧数值基线。必须在每个外层任务中重新执行固定 12 epoch 的 RGB 源域 R18 训练，保存权重与 prefix，再分别训练 aligned_only/joint 两个六 epoch layer4。所有目标域图像只用于冻结后特征计算，目标标签不进入编码器训练。SAR checkpoint 不参与 RGB 任务；不增加特征 MLP 冒充 layer4。

## RGB 阶段实现

脚本：`E:/Hermes/paper_tracking/sensitivity/oof_reliability_check/generic_joint_completion.py`。

输出隔离于本目录 `rgb_retrained/{dataset}/{domain}/`；已有 M2 结果不覆盖。

1. `--stage images`：按 parquet 原顺序流式提取编码图像，一次 32 条；不整体载入图像表。
2. `--stage resize`：一次性生成 deterministic PIL bilinear Resize224 的 uint8 memmap；CPU 每批32条顺序写文件，不整体载入图像表。
3. `--stage bootstrap`：每个外层域排除目标域，ImageNet-V1 RGB R18，Adam lr=0.001，12 epoch，batch=64，固定新种子；GPU 随机水平翻转和归一化、bf16 autocast、FP32模型/Adam。保存完整 R18 与 float16 layer3 缓存。八个外层域使用同一冻结吞吐协议。新初始化特征也由同一个量化 prefix 生成，供四臂匹配。
4. `--stage prepare`：CPU 两线程拟合三组 inner-source 头与 OOF 门控，构建源域 64-support/query episodes，保存固定源 PCA、补空间统计量与部署头。编码器不是严格嵌套 OOF，这一点与旧 generic 协议相同且显式标注。
5. `--stage train`：仅 RGB R18 layer4；源查询 CE 经真实 64-support ridge8 回传，六 epoch、AdamW lr=1e-5、weight_decay=1e-4、clip=5；两臂相同初始化和批次顺序，BN 运行统计冻结，最后一轮模型用于部署。
6. `--stage evaluate`：四臂共用新源参考、原目标支持集、原 base64 得分、同一新源门控；固定源几何不在训练后重拟合。先保存四臂全部策略，再评估 OA/BA。输出明确 `old_reference_exact_replay=false`，不可混入原旧基线作为精确重放。

下面是单域执行顺序示例，不是已运行记录：

```powershell
$cpu = 'D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe'
$gpu = 'E:/Hermes/paper_tracking/envs/msfa-openmmlab-py310-rtx50-cu132/Scripts/python.exe'
$script = 'E:/Hermes/paper_tracking/sensitivity/oof_reliability_check/generic_joint_completion.py'
& $cpu $script --stage images --dataset pacs --threads 2
& $cpu $script --stage resize --dataset pacs --threads 2
& $gpu $script --stage bootstrap --dataset pacs --domain art_painting --device cuda --batch-size 64 --threads 2
& $cpu $script --stage prepare --dataset pacs --domain art_painting --threads 2
& $gpu $script --stage train --dataset pacs --domain art_painting --device cuda --threads 2
& $cpu $script --stage evaluate --dataset pacs --domain art_painting --threads 2
```

GPU 依赖修复已完成：CPU 几何/评估相关导入改为函数内延迟导入，bootstrap/train 只读取已导出的图像元数据，不再导入 `ot` 或 `pyarrow`。CUDA 环境 `--help` 及模块导入检查已通过，无需额外安装。两个数据集编码图像均提取完成；PACS 的 uint8 Resize 缓存已完成，用时51.3秒。

## 排期和资源

- M2 已完成且进程退出。旧 FP32 PIL 路径的首个 bootstrap 在完成三个 epoch 后按主代理要求停止；旧工作 PID 23828 已释放 GPU。partial 保留在 `partial_runs/pacs_art_painting_fp32/`，没有目标查询评估结果。三轮耗时73.27/82.75/72.78秒；不作为正式十二轮参考。
- 用户授权所有八折在读取新目标结果前统一采用缓存与 bf16 的吞吐优化。新 reference 必须从 ImageNet 初始化重新训练十二轮，不续接旧 FP32 partial。GPU 后续启动仍由主代理排期。
- 完整训练仍固定十二 epoch；`--bootstrap-stop-after 1` 可在第一 epoch 保存包括优化器和 RNG 状态的断点后退出，后续不带该参数会继续完成余下 epoch，不改为一 epoch 方法。
- PACS 与 Office-Home 全八个外层域的源编码器12 epoch、M3/joint各6 epoch及评估均已完成，无剩余 GPU 训练排期。
- 逐域单进程即可运行；原图流式提取约 1.35 GB 磁盘，float16 256x14x14 prefix 每 PACS 外层约 1.0 GB、每 Office-Home 外层约 1.56 GB。八个域合计约 10.27 GB prefix，另加约 0.38 GB R18 权重及特征/episodes，建议预留约 15 GB。
- 原图不全部解码驻留，训练按批 64 读取；需要给 PyTorch、批数据和源 PCA 工作数组留出数 GB 可用 RAM。资源紧张时不同时启动第二个大型 CPU/图像过程。
- 已完成全八域旧参考上的 M2（80条），以及 PACS/Office-Home 全八域新参考下的四臂完整训练和评估（各80条，共160条）。两种协议输出分离，负结果完整保留，全部 GPU 资源已交还主代理。
