# 强单骨干本地候选审计

后续执行更新：本文件保留启动前的资产审计状态。当前已完成SARCLIP原生双极化的24港/OSS2冻结筛查，以及OSS2单R50三个种子的任务式训练，见同目录PROGRESS_ZH.md。SARCLIP当前环境运行不再待验证；F_0921新对照及预训练暴露边界仍未解决。

日期：2026-10-10。独立只读支线；只新增本文，未训练、未跑GPU、未下载大权重、未安装或修改主脚本。已读 brain08、CRCA skill 头部及同目录 `FAIR_COMPARISON_ZH.md`。

## 结论先行

**现有证据不能让双骨干直接通过强单骨干准入。** 可以立即着手的资产包括 ImageNet R50、SAR SSL-R18、已有CUDA运行记录的 SARCLIP ViT-L/14，以及备用 EfficientNet-B4。SARATR-X 只有源码和论文，没有在本次核查范围找到权重；当前匿名访问仍被门控。模型名和文件存在均不证明当前八类任务上的强度。

最少应覆盖：任务目标对齐的强R50、一个SAR预训练单骨干、任务导向单骨干蒸馏，并以实际成本判断是否还需容量/预算相近的单模型。外部预训练性能竞争与严格同信息架构归因分表，不能互相替代。具体准入规则以同目录正式协议为准。

## 1. 实际资产与可执行程度

| 候选 | 已证实的实际路径/版本 | 输入与运行状态 | 本轮判断 |
| --- | --- | --- | --- |
| R50 ImageNet V2 | `C:/Users/25791/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth`，102,540,417 bytes；CPU只读加载成功，原始conv1为64x3x7x7 | 当前项目转VV/VH双通道，卷积权重为RGB均值重复两路再乘1.5；128输入，`(x/255-.5)/.25`，C4四次共享权重前向。原图与C4等多层/多头不应被禁止给单骨干使用 | 主线程同任务训练的直接起点；冻结删除对照不是强单的终点 |
| 已有R50全参数CE试验 | `E:/Hermes/paper_tracking/sensitivity/completion_single_finetune_20261010/r50_final.pt`，94,713,483 bytes；本轮CPU读取容器成功，键为state/source_rows/seed/epochs | OSS2允许源行36,996、8轮、一个神经种子；目标同5组64支持；简化C4读出，不是通过最终支持残差目标训练 | 可追溯的旧检查点，不能用其负结果排除强单；不直接跨外留任务复用 |
| e206 SAR SSL-R18 | `E:/临时会话/visual_reliable_baseline/e206_out/ssl_resnet18.pt`，44,784,207 bytes；本轮CPU加载成功，epoch=40 | 随机初始化SimCLR/NT-Xent，灰度PNG复制RGB，训练裁为112；当前部署为VH、0..1、128相关变换，无ImageNet归一化。OSS2部署保留原`Resize(128)`长宽比路径，不应擅自当统一128方图 | 可立即构成“同现有SAR初始化的R18单路”归因控制，但R18容量小，不能单独代表所有强单 |
| SARATR-X HiViT | 源码 `E:/Hermes/paper_tracking/repos/SARATR-X`，本地HEAD `e1eb57f2881596a587f60898cb20e7f7372eb9ca`；本次相关目录未发现预训练权重 | HiViT类在现有项目环境CPU导入成功；Dassl不存在；完整官方分类入口未运行，未加载权重或forward | 明确阻塞在权重及完整适配，不是已完成/可立即运行的基线 |
| SARCLIP ViT-L/14 | `E:/Hermes/paper_tracking/models/sarclip/BiliSakura_SARCLIP-ViT-L-14/model.safetensors`，1,710,526,476 bytes，文件现在存在；来源是HF镜像BiliSakura，而非据名字假定作者原始发布文件完全等同 | 现成后端 `E:/Hermes/paper_tracking/prototypes/caiss_sar_ship/sarclip_hf_vitl14_backend.py`；224三通道灰度相关输入、[-1,1]、768维。历史真实CUDA试验完成446/446键转换、1735图提特征；不是仅随机模型smoke | 最直接的已落盘外部SAR预训练竞争者；还没有本轮三数据集八类同64支持成绩；预训练样本暴露及当前环境依赖待核实 |
| EfficientNet-B4备用 | `C:/Users/25791/.cache/torch/hub/checkpoints/efficientnet_b4_rwightman-23ab8bcd.pth`，78,107,389 bytes；CPU加载成功 | torchvision `IMAGENET1K_V1`；原生resize384/crop380、RGB ImageNet归一化，非当前R50双极化128配方；未做当前任务forward/训练 | 可作不同单架构的本地备用，先配输入信息与实测成本；不能凭参数或名字宣称更强、等计算 |

另有 ImageNet R50 V1 `C:/Users/25791/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth`、R18 `resnet18-f37072fd.pth`、EfficientNet-B0 `efficientnet_b0_rwightman-7f5810bc.pth` 的真实缓存，但不建议靠堆同类弱对照增加表格行。

SAR检测资产也存在，例如 `E:/Hermes/paper_tracking/checkpoints/MSFA/ckpts/fg_frcnn_dota_pretrain_sar_wavelet_r50/best_coco_bbox_mAP_epoch_12.pth`。它是检测checkpoint，不是已经适配好的八类舰船单骨干，故不加入最小集合。没有新增FUSAR、OSS1或通用跨域实验。

## 2. SARATR-X：权重、源码与OSS2暴露

### 可用性

- 核查范围为官方本地仓库、项目checkpoints、SARCLIP模型目录、hf_home/cache_home及用户默认HF缓存；未找到SARATR-X的checkpoint/snapshot。不声称穷尽所有磁盘或改名文件。
- 本轮官方HF模型API返回200：`private=False, gated=auto`，版本 `32cc9ecca6c33832579b189221841cc52e4d45e3`；`weight/186K_notest/checkpoint-600.pth` 匿名HEAD返回401、`GatedRepo`。未下载正文。
- 同日既有核查记录：notest约1,044,098,904 bytes；all/checkpoint-200约1,044,098,136 bytes；已有凭据访问曾返回403。本轮未重复读凭据，不能把历史403说成新鉴权结果。百度备选只有入口记录，未验证实际下载。
- 当前解释器 `E:/Hermes/paper_tracking/envs/msfa-openmmlab-py310-rtx50-cu132/Scripts/python.exe` 可导入 `classification/model/hivit.py`，但缺Dassl；导入成功不是权重加载、完整官方入口或推理成功。

### 实际分类实现

`E:/Hermes/paper_tracking/repos/SARATR-X/classification/trainers/MIM_linear.py` 使用HiViT embed_dim=512、depths=[2,2,20]、8 heads，不是文件名`vit_b16`暗示的普通ViT-B。单通道在forward复制三通道，硬编码加载`checkpoint-200.pth`中的`model`，处理`module.`前缀及位置嵌入，冻结主体但训练fc_norm和BN+分类头。加载使用strict=False，取得权重后必须检查缺失/多余关键层，不能静默接受随机主体。

配置 `E:/Hermes/paper_tracking/repos/SARATR-X/classification/configs/trainers/MIM_linear/vit_b16.yaml` 为224、bicubic、TRANSFORMS空，30轮AdamW lr1e-3、weight_decay1e-4。空TRANSFORMS不能单独证明全部数据强度转换；还需核对具体数据读取器。若使用统一ridge头，必须标明“官方预训练特征的协议适配”，不是完整官方LP复现。取得notest权重后也不能原封不动依赖脚本中all版本文件名。

### 预训练重叠，不能只写“可能同系列”

本地官方论文文本 `E:/Hermes/paper_tracking/papers/saratr-x_2405.09365.txt` 的预训练清单列出OpenSARShip 26,679片；文末参考文献[79]明确是 **OpenSARShip 2.0: A large-volume dataset...**。因此它确实使用了OSS2数据来源，不能把它当作已证明未接触OSS2的外部预训练。

原文脚注5称移除其下游测试集，但没有给出本项目当前3745条目标样本的交集明细。`186K_notest`不等于“排除我们这5组64支持及全部查询”。**数据集级预训练暴露已知；本项目具体裁片/场景交集数未核实；不据无标签暴露推断目标类别标签泄漏。**

自建24港/F没有在官方清单中以这些名字出现，不代表原始Sentinel-1场景必不重合。外部竞争可以单列“公开预训练、暴露待核实”，不能用于严格同信息架构归因。

## 3. e206实际训练与三个数据集的暴露边界

这次不只读取脚本默认值，而核对了训练日志与实际checkpoint：

- `E:/临时会话/visual_reliable_baseline/e206_ssl.txt` 开头记录语料345,590片；末尾记录ep40/40完成，产物正是 `e206_out/ssl_resnet18.pt`。
- `E:/临时会话/visual_reliable_baseline/e206_out/pretrain_log.txt` 记录40轮完成；本轮只读加载当前checkpoint的epoch也是40。日志、代码和产物一致，支持此训练谱系，但权重本身只存backbone/epoch，没有完整训练样本ID清单。
- `e206_ssl_pretrain.py` 实际代码递归读取 `E:/F_0921_pool_v1/**/*.png`，无按目标港/MMSI排除逻辑。不能把后续有监督LOPO排除倒推成SSL也做了排除。
- `e207_ssl_dchips.py` / `e208_ssl_cached.py` 确实另写了24港dchips语料，但可执行OUT是 **e207_out**；头部沿用的“e206_out”注释已过时。不能据这些注释认定当前e206权重是在24港池重训，更不能拿e207的干净语料说明替换e206暴露边界。

| 评价数据 | 能确认到的层级 | 仍不能确认 |
| --- | --- | --- |
| F_0921 | e206使用该来源的无标签池；不是ImageNet-only初始化。池的`labels.csv`含2021年北美SAR/AIS记录 | 与本轮123489查询候选逐裁片、场景、MMSI交集未计算；“同来源无标签预训练”不等于每条查询都参与训练 |
| 24港 | 池的 `E:/F_0921_pool_v1/domain_merged_counts.csv` 包含Los Angeles/Long Beach 34770、Houston 19957、New York/New Jersey 10238；当前24港manifest也有Los Angeles/Houston/New York外留任务。因此这些目标港**港域级无标签暴露**不能忽略 | 同港不等于同场景/同船/同裁片；具体交集未核实，其余港不能仅由表面名称判完全无暴露 |
| OSS2 | 未发现e206训练代码直接读取OSS2的证据，现有日志是345590片F池训练 | 尚无全量身份/场景交集排除证据，不能标“已证实零暴露”；这与SARATR-X已知使用OSS2来源是不同证据等级 |

港口汇总表共345587条，较训练日志345590少3条；它适合证明已记录的港域覆盖，不可当成与训练PNG一一对应的完整训练manifest。本文没有为补这3条扩张成全量追溯任务。

因此共同无标签访问或源端预训练的对齐规则应覆盖**每一个实际评价任务**，不仅F。还应对齐ImageNet初始化、SAR语料访问及源支持残差训练机会，不能只给单路同样64标签就宣称完全公平。

## 4. SARCLIP的可复用价值与缺口

以较晚完成的 `E:/Hermes/paper_tracking/run_logs/2026-06-29_sarclip_full_cuda_sync/README.md` 及 `E:/Hermes/paper_tracking/run_logs/2026-06-29_sarclip_physical_leavebin_full_cuda/report.md` 为准：历史2.10.0+cu128环境已完成真实CUDA键转换/1735图特征及旧5类双向试验。早期CPU-only或CUDA故障记录不能覆盖后续成功记录。

但这些结果不是本轮23港/F/OSS2八类结果，不引入旧FUSAR成绩参与当前排名。现有后端依赖少量导入stub，且模型是HF镜像键转换版本；在新训练实现中应复用已验证视觉映射而清楚标记移植边界，不把stub文本分支称为有效文本推理。当前cu132环境兼容性本轮未重测。

SARCAP及镜像checkpoint的完整预训练样本/场景清单没有在本次只读核查中得到，F、24港、OSS2的暴露均未排除，不能因模型叫SARCLIP就当作严格同信息控制。视觉单路只输入图像，不额外给予AIS/类别描述检索信息；输入极化方案需源端预先固定，不按目标结果挑VH/VV/融合方式。

## 5. 最小有说服力集合与未完成工作

1. **主线程的强R50同任务训练**：正确双极化输入；允许多层/多头，训练通过最终支持/查询任务目标；源验证选学习率、轮次、尺度，多个神经种子。现有8轮CE仅是旧参考，不能排除这一对手。
2. **一个SAR预训练单骨干**：当前优先候选SARCLIP，因为文件和旧CUDA证据真实存在；SARATR-X权重到位后可纳入预先固定的外部集合，不看目标表现后替换“赢家”。现有e206-R18单路可补训练来源控制，但不能替代容量足够的外部强单。
3. **任务导向R50蒸馏**：学习最终类别证据/支持关系而非只做坐标MSE；计入教师训练费用，部署只保留单骨干。旧MSE映射失败不是该对照的替代品。
4. **条件补一个计算匹配单骨干**：若上述单骨干均不在双路附近的成本区间，再使用已缓存B4或其他已核实候选；先测真实成本，不预设B4就是等预算。无需为凑数量立即扩成大模型榜单。

仍未满足：SARATR-X权重加载及完整输入链；本轮外部SAR单骨干八类统一评价；e206及外部模型的逐任务预训练身份交集；充分且对称的任务训练/源验证预算；多神经种子；对选定强单的净救回/误改和实际精度-成本比较。已有冻结R2D2头、删R18、8轮CE、MSE重建、旧SARCLIP5类成绩均不能代替这些条件。

**目前只能说“第二路对既有受限对照有增益”，不能说“强单做不到”。** 只有预先固定、源端选定的实际强单集合完成后，才分析双路独有救回；该结论也应限定为“已测试集合未救回”，不是所有可能单骨干理论上不能救回。

## 核查边界

本次检查仅用于区分真实资产、过时阻塞记录和公平性缺口。CPU检查点读取/HiViT导入进程均已退出；未运行推理性能试验。官方联网仅做HF小元数据与匿名HEAD，未申请访问或改变账号权限。详细旧访问记录在 `E:/Hermes/paper_tracking/sensitivity/frozen_comparison_20261010/PUBLIC_METHODS_ZH.md` 第6节，不能把该旧节的所有动作算作本轮新执行。
