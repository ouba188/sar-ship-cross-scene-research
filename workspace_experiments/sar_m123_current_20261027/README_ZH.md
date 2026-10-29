# 当前骨干内 M1/M2/M3：完整训练与预测工程

2026-10-29结构收口：M1、M2去掉频域分支，M3保持原设计。见 [SPATIAL_M12_CLOSEOUT_ZH.md](SPATIAL_M12_CLOSEOUT_ZH.md)；工程烟测见 `verification/spatial_m12_20261029.json`。新结构未做配套训练与精度验证，下列历史结果不能直接沿用。使用新架构需显式加载迁移后的新checkpoint。

已完成的实际验证见 [VERIFICATION_ZH.md](VERIFICATION_ZH.md)：四模型原图重放、136条干预选样一致、完整6轮适配重跑和160条预测一致。初始8轮源拟合没有重新全量训练，仅完成真实单步与梯度验证。

本版对应2026-10-27干预选样＋原教师单侧分差约束候选。不是早期末端 M1/M2/M3，也不是5%专家交接模型。保留既有正式模型，不自动替换默认配置。现按用户要求同步完整运行源码与验证记录到GitHub。

## GitHub发布范围

本目录包含完整模型、fit/select/adapt/predict入口、单元测试及小型验证记录；不包含权重、原始chip、实际样本清单和大型缓存。下文的`weights/`、`images/`、`manifests/`及逐样本验证文件属于本地完整交付，不能仅克隆仓库就运行真实数据重放。请准备自己的合法数据清单与完整checkpoint，或使用本机`E:/临时会话/sar_m123_current_20261027/`已有资产。数据格式与预处理以`current/data.py`、`input.py`为准。

不依赖历史实验脚本是指运行代码独立，并不意味着无需数据或训练权重。此次同步没有新增训练或泛化结果。

## 干预选样在哪里

它是**训练前的源数据处理步骤，不是M4，也不是预测门控**。

```text
阶段A：源数据拟合
源VV/VH chip + 固定源参考 -> 共享R50与极化专属M1/M2/M3 -> 分类训练 -> 冻结教师

阶段B：训练专属选样
源适配样本原图/受控干预/配对均值控制 -> 同一冻结教师 -> 比较NLL及类别分差
  -> 选出结构变化确有收益的干预
  -> 固定显式频率为原图，再复核图像路径收益 -> useful标记

阶段C：局部适配
所有源适配样本原图 -> 学生 -> 全部样本CE
                            + 原教师判对样本KL
                            + useful样本的原教师单侧分差下界
只更新共享layer3.5.conv1/2/3；M1/M2/M3、分类头、BN、其他卷积冻结。

预测：原始VV/VH chip + OBB + 固定参考 -> 完整网络 -> 八类logits/概率/预测
无干预，无查询标签，无选样器，无测试标签调参。
```

136条适配样本中101条被选中是历史结果，不是硬编码数量。没有被选中的35条仍参与分类训练。参考标签用于初始关系监督；当前M3预测前向并不读取参考类别标签。

## 真实网络路径

两种极化分别输入，共享同一组ResNet-50卷积权重，不是两套骨干。M1/M2/M3各有VV和VH参数。

```text
原生8-bit chip -> OBB定位中心裁剪/补零160，禁止缩放
  -> shared Stem/layer1/layer2 -> layer3前5个block
  -> layer3.5: conv1 -> BN/ReLU -> M1 -> M2池化回写 -> conv2 -> conv3 -> 残差
  -> M3空间—频域参考匹配回写 -> layer4 -> 有效区域全局均值
  + layer2与M3后的layer3区域池化读出 -> 拼接两极化描述 -> 分类头
```

- M1：类别响应均值/高侧/低侧注意力写回。
- M2：框内、轴端、侧向分开池化并回写；另保留多层区域读出。并非替换所有原生池化。
- M3：冻结比较锚点与当前可训练value路径分开；同区域局部匹配、未匹配处理、参考聚合和残差写回。
- 频域是原生显示chip的4/8/16像素Gabor能量，不是SLC相位或电磁频谱。干预使用Gaussian差分频带，两者不能混称。
- 不额外输入雷达元数据、AIS航向、港口地理先验。OBB给出图像船轴；MMSI仅在已知时排除同船参考。
- 未证明增益的联合极化条件头，在当前权重中增量恒为零，导出时删除；其余有效权重全部保留。

## 文件导航

| 文件 | 用途 |
|---|---|
| `sar_m123/current/cli.py` | predict/select/fit/adapt统一入口 |
| `sar_m123/current/data.py` | 原始chip、OBB、有效性、空间槽位与频率输入 |
| `sar_m123/current/model.py` | 完整权重加载、前向、共享卷积训练策略 |
| `sar_m123/current/interventions.py` | 干预生成、配对控制、两阶段源选样 |
| `sar_m123/current/training.py` | 初始模块拟合、干预选样单侧适配 |
| `sar_m123/bottleneck_evidence.py` | M1/M2在残差块内的真实插入位置 |
| `sar_m123/selective_readout.py` | M1及M2的证据读出算子 |
| `sar_m123/selective_matching.py` | M3比较、匹配和回写 |
| `sar_m123/selective_backbone.py` | 参考分支、多层读出和分类路径 |
| `weights/init.pt` | 与历史源拟合相同的初始化，尚未完成源训练 |
| `weights/teacher.pt` | 完整冻结教师，原baseline17固定第8轮 |
| `weights/selected_s17.pt` | 完整适配后模型，默认演示单模型 |
| `weights/selected_s29.pt`, `selected_s43.pt` | 其余两个训练顺序种子，不是集成 |

运行时不依赖原实验run.py、特征缓存、历史import链或原预训练权重路径。完整checkpoint包含R50、冻结锚点、M1/M2/M3及分类头，不只是adaptation.pt增量。

## 直接运行

在本工程目录执行；本机Python可用 `D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe`。CPU可加`--device cpu`。依赖见pyproject.toml，已有环境无需重复安装。

```powershell
python -m sar_m123.current predict --model weights/selected_s17.pt --manifest manifests/demo.json --references manifests/references.json --output outputs/demo.json
python -m sar_m123.current select --model weights/teacher.pt --manifest manifests/source_adapt.json --references manifests/references.json --output outputs/selection.json
python -m sar_m123.current adapt --model weights/teacher.pt --manifest manifests/source_adapt.json --references manifests/references.json --selection outputs/selection.json --output outputs/adapt_s17 --seed 17
python -m sar_m123.current fit --model weights/init.pt --manifest manifests/source_fit.json --references manifests/references.json --output outputs/source_fit --seed 17
python test_current.py
python verify_current.py --selection all
```

若从头重新拟合，后续select/adapt的`--model`都应使用新fit输出；**不能把旧教师的选样文件直接用于新教师**。当前默认指导系数0.3683066666来自历史源数据梯度标定，重做不同数据实验时需在源端另行固定，不能根据测试涨点选取。

初始fit：8轮，batch24，模块lr0.001，共享局部卷积lr0.00003；CE＋0.1类别响应辅助＋0.1参考关系监督。适配：6轮，batch16，AdamW lr0.00003，weight_decay0.01；固定末轮，不从验证集挑最好轮。

当前适配损失：

`L = weighted_CE + 0.1 * correct_teacher_KL + 0.3683066666 * selected_margin_floor`

`floor = average_{c != y} relu((teacher_y-teacher_c) - (student_y-student_c))^2`

floor只在选中样本非零，所有项以整个batch权重和归一化。学生超越教师不罚。这里教师指**原图教师**，不使用干预后的分差作为强制目标。训练固定BN，但不能用no_grad包住可训练卷积与后续模块；参考的live特征每批随学生重算。

## 输入格式与数据

manifest是JSON列表。每条具有`id`，可选`mmsi`；训练另需`label`、`split`。预测不需要label。VV/VH独立描述，缺失或不可靠的一侧设null。

```json
[{"id":"sample-1", "mmsi":"optional-vessel-id", "split":"source_adapt", "label":0,
  "vv":{"image":"chip_vv.png", "target_index":0,
        "detections":[{"corners":[[40,40],[80,40],[80,55],[40,55]],"confidence":0.9,"shape_usable":true}],
        "ship_axis":{"center":[60,47.5],"length_px":40,"width_px":15,"angle":0}},
  "vh":null}]
```

OBB角点和船轴中心对应**原始chip像素坐标**；角度弧度，x向右、y向下。干预需要可靠ship_axis；正常预测可由OBB推导。当前只接受沿用实验预处理的8bit显示图，不能直接把原始辐射强度TIFF当PNG输入。目标超出160画布时报错，不偷偷缩放截断目标。

参考库16条及演示chip已复制进images/，路径相对manifest可搬移。完整1205训练、136适配和160开发manifest指向本机已有数据，没有重复复制全部数据集。搬到其他电脑需同步这些数据并修改manifest图像路径。模型代码、完整权重与小演示可独立搬移。

## 证据和验证边界

当前扩围3075条：适配版相对初始教师OA/BA提高约0.77/2.30个百分点；相对同类配额随机选样，额外OA/BA近似持平，NLL/Brier更好。不能把完整适配收益全部归因于干预选样。

工程整理的成功标准是：从原始chip重建输入后，输出与历史缓存一致；选样标记可重算；反向梯度到达正确参数；checkpoint可完整重载。详见verification/receipt.json。这不是一轮新泛化实验，也不增加论文独立证据数量。

原干预研究使用已反复探索的源开发数据。所称干预证据是模型对受控图像变化的响应，不等于已分离纯物理因果因素。预测softmax输出未经额外概率校准。
