# 当前 OSS2 工作版的载体与 M3 分阶段训练依据

日期：2026-11-09。范围：只读追溯已选工作版上游；只新增本文。不训练、不调用 GPU、不修改源模型或源码，不审查新 staged.py。以下给出已核实的可用配方；停止继续扩展历史。

## 1. 可立即用于实现的阶段表

| 阶段 | 查询样本 | epoch / batch / steps | 可训练参数 | AdamW 学习率 | 损失及端点 |
|---|---:|---|---|---|---|
| 历史载体 source_fit | 478 | 8 / 24 / 160 | layer3[5] 的 conv1/2/3.weight；旧空间 M1、M2、旧 M3、head，排除冻结 reader | 三个卷积 3e-5；其余模块/head .001；wd=.01 | 船/类加权 CE + .1 M1辅助CE + .1 M3关系损失；固定第8轮 |
| 历史载体 source_adapt | 241 | 6 / 16 / 96 | **仅** layer3[5] 的 conv1/2/3.weight，1,114,112 参数 | 3e-5；wd=.01 | 类平衡 CE + .1 correct-teacher KL + .3683066666126251 selected margin floor；固定第6轮 |
| M3 aggregate 替换 | 719=478+241 | 4 / 24 / 120 | **仅**新 matcher 和载体 head，共96,828参数 | head 1e-5；matcher.output 1e-4；其余 matcher 1e-3；wd=.01 | 加权完整CE；clip_grad_norm=5；当前继承固定第4轮 last.pt |

载体 fit/adapt 没有学习率调度或梯度裁剪调用；M3 有 clip5。三段每次新建优化器，不继承 Adam 状态。batch 最后不足额仍参与更新，不 drop_last。网络保持 eval，BN 统计与 affine 固定。历史 fit/adapt 的 seed17/29/43 只改变顺序，共用 seed17 初始化；当前工作版实际继承的是 full_s17。

直接依据：

- [历史 protocol.py](E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/final_oss2_replication_20261029/protocol.py:10)：第10行初始化，第12行类映射，第18至20行数量/预算/系数，第42至55行权重。
- [历史 run.py](E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/final_oss2_replication_20261029/run.py:137)：第137至219行完整训练过程；153至156行分组LR，179至188行两种损失，199至210行固定端点保存。
- [training_policy](E:/临时会话/sar_m123_standalone/sar_m123/current/model.py:50)：冻结策略；LIVE 名称在第13行。
- [M3 train_task.py](E:/临时会话/m3_frequency_increment_20261028/source_hard_task_20261101/oss2/train_task.py:28)：第28至39行参数与权重，第45至68行4轮/24batch/CE/clip/last，第74行另存 selected。
- 实际收据：[fit/training.json](E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/final_oss2_replication_20261029/training/full_s17/fit/training.json)、[adapt/training.json](E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/final_oss2_replication_20261029/training/full_s17/adapt/training.json)、[aggregate_17/receipt.json](E:/临时会话/m3_frequency_increment_20261028/source_hard_task_20261101/oss2/aggregate_17/receipt.json)。fit 可训练参数2,462,276；adapt1,114,112；aggregate96,828。

## 2. 必须保留的 intervention adapt：准确实现与路径

### 2.1 数据、教师、缓存

历史公开载体根目录记为 P：

`E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/final_oss2_replication_20261029`

| 对象 | 准确文件/目录 | 用途 |
|---|---|---|
| fit样本 | P/manifests/source_fit.json | 478张 |
| adapt样本 | P/manifests/source_adapt.json | 241张；全部参加CE |
| 固定参考 | P/manifests/references.json | fit内32张，每类8张 |
| 本模型教师检查点 | P/training/full_s17/fit/model.pt | 生成自己的原图/干预分数 |
| 已有历史教师记录 | P/teachers/full_s17.json | 每条 original、altered、useful、choice、selected_intervention |
| 教师收据 | P/teachers/full_s17.receipt.json | 教师路径及136条 useful |
| 原图输入缓存 | P/cache/*.npy，P/cache/complete.json | float32 prefix及掩膜/极化/空间频域输入；非训练后特征缓存 |
| 适配端点 | P/training/full_s17/adapt/model.pt | 固定96步，非验证选优 |

上述 P 是路径缩写，不是环境变量。新分阶段实验必须在自己的输出目录保存各 arm/seed 的 fit 教师、选择记录和适配端点；**不能直接使用 full_s17.json 中的教师分数或136个标记去监督新模型**。136/241是历史结果，新模型应按同规则重新选样，实际数量可以不同。

缓存生成：[run.py:34](E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/final_oss2_replication_20261029/run.py:34)。按8条提取冻结 layer2 prefix，形状 `[1519,2,512,20,20]`；其余键为 `masks,valid,available,slot_masks,frequency,local_slot_masks,local_frequency,field2,field3`。读取与前向见同文件第74至102行。当前新实现已有合法 early-prefix 缓存，不要求复制历史大缓存；但所有干预版本都必须由对应的修改图像重算被冻结前缀，不能用原图 prefix 冒充修改图像。

### 2.2 精确目标函数

[training.py:15](E:/临时会话/sar_m123_standalone/sar_m123/current/training.py:15) 定义 `margin_floor`；第25行定义 `adaptation_loss`：

```text
w_i = 241 / (4 * source_adapt中类别y_i的样本数)
CE = sum_i w_i * CE(student_i, y_i) / sum_i w_i
correct_i = argmax(original_teacher_i) == y_i
keep = sum_i w_i * correct_i * KL(teacher_i || student_i) / sum_i w_i
teacher_margin_ic = teacher_i[y_i] - teacher_i[c]
student_margin_ic = student_i[y_i] - student_i[c]
floor = sum_i w_i * useful_i * sum_c relu(teacher_margin_ic-student_margin_ic)^2 / 3 / sum_i w_i
L = CE + 0.1*keep + 0.3683066666126251*floor
```

teacher全程detach；KL温度1。floor以**原图教师**分差为下界，历史 `adaptation_loss` 没有将 altered 传给 `margin_floor`；因此不是蒸馏被干预图像的更好分差。useful只控制floor，正确教师标记控制KL，所有241张都有CE。不能将adapt简化成仅CE、只训练136张，或误写成altered教师优势蒸馏。

历史调用在 [run.py:161](E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/final_oss2_replication_20261029/run.py:161)：教师记录逐条对齐id/label，第183至188行使用original/useful。第一步要求当前原图logits与自己的teacher接近，以发现教师/参考/输入不对应。

### 2.3 原图干预选样

- 入口：[run.py:223](E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/final_oss2_replication_20261029/run.py:223) 的 `select_one`，调用自己fit检查点和固定参考，读取source_adapt。
- 实现：[interventions.py:28](E:/临时会话/sar_m123_standalone/sar_m123/current/interventions.py:28) 的 `variants`：逐VV/VH、body/side、fine/middle；sigma1/2高斯差分，区域可见像素至少8，边界taper；共同剂量为候选范数最小值的0.5；每个结构修改配套mean_control。返回原图及修改后的图像输入。
- 筛选：[interventions.py:71](E:/临时会话/sar_m123_standalone/sar_m123/current/interventions.py:71) 的 `select_teacher`：原图最强竞争类固定；CE改善、相对原图分差改善、相对mean_control分差改善均至少.01；符合者按最后一项最大选取。
- 完整重算：[interventions.py:97](E:/临时会话/sar_m123_standalone/sar_m123/current/interventions.py:97) 的 `select_records`，修改图像每8条前向；第118至119行再要求真实类log概率与相对实际最强竞争类分差均优于原图。
- 图像前缀重算：[current/model.py:76](E:/临时会话/sar_m123_standalone/sar_m123/current/model.py:76)。该旧入口要求旧模型接口；**新 staged.py 的网络不能仅因函数名相同就直接传入**，需用新模型真实forward重放同一选样规则。这里未审查或实现该适配。

旧plain的useful全False、coefficient=0，仍有CE+.1正确教师KL；见历史run.py第165至167、238至244行。新最终存活算子实验如何给各消融臂实施同规则干预、SE臂是否参与相同训练目标，需在新协议中明确，不能借历史plain约定默认省略所有臂的干预阶段。

## 3. 已核实的上游权重链与最早起点

```text
ImageNet resnet50-11ad3fa6.pth + fresh task modules/head
  -> sar_m123_current_20261027/weights/init.pt
  -> 四类切片 + SpatialM12 结构（去旧M1/M2频域，旧M3仍含频域）
  -> P/training/full_s17/fit/model.pt               160 steps
  -> P/training/full_s17/adapt/model.pt              96 steps
  -> 旧M3关闭频域，构建固定空间载体输入；替换为新aggregate matcher
  -> m3_frequency_increment_20261028/source_hard_task_20261101/oss2/aggregate_17/last.pt
                                                     120 steps，保存head+matcher增量
  -> m1_coverage_20261101/small_class_s17/final.pt      M1阶段由主代理负责
  -> m1_guided_m2_20261106/class_s{17,29,43}/final.pt   末阶段由主代理负责
```

根指针：[current_version.json:9](E:/临时会话/m1_guided_m2_20261106/current_version.json:9)，M3路径/缓存执行说明在第19至22行。[model_task.py:61](E:/临时会话/m3_frequency_increment_20261028/source_hard_task_20261101/oss2/model_task.py:61) 明确载体从full_s17/adapt加载且关闭旧M3频域。[run_pilot.py:26](E:/临时会话/m1_role_separation_20261101/run_pilot.py:26) 明确anchor恢复aggregate_17/last.pt，并非selected.pt。

**init.pt 不是已训练的八类SAR检查点。** 本次只在CPU读取其metadata：`architecture=native160_pre3x3_m123`、`source=exact source-fitting initialization seed17`、`training_stage=initial`。直接比较同机 `C:/Users/25791/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth`：320项 `backbone.model.*` 全等，18项 `anchor_last.*` 与 ImageNet layer3.5 全等；head形状8x7170，其两段区域列非零数为0；CUDA未初始化。

构造来源也支持没有SAR训练权重进入init：

- [export_current.py:115](E:/临时会话/sar_m123_standalone/tools/export_current.py:115) 调 `prior.make('baseline',17)` 后导出init；不是第104至112行导出的teacher/selected。
- [joint_polar_m1/run.py:67](E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/joint_polar_m1_20261027/run.py:67) 从 `bottleneck.make` 新建，不调用restore。
- [bottleneck_internal/run.py:26](E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/bottleneck_internal_20261027/run.py:26) 从 `selected.make` 复制新建状态，再新增内部M1/M2。
- [selective_m123/run.py:103](E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/selective_m123_20261027/run.py:103) 显式新建seed控制的模块/head；没有加载训练后模块。布局来自该目录 `selection.json` 的all_deep2。这说明**结构选择用过历史开发数据，但不等于初始化权重做过源训练**。
- [final_module_ablations/model.py:95](E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/final_module_ablations_20261029/model.py:95) 将init状态按新结构同名拷贝，head及M1 reader按 `[0,2,3,4]` 切片。

无需将自建1205/136样本的teacher/adapt训练加入当前OSS2权重链；那些是export的另两个产物，不是init的上游训练。本文没有做全历史审计，也未声称模型设计从未参考其他开发集。

## 4. M2载体到底训练了什么

旧公开载体是 `SpatialM12ResNet50(pre3x3, local, classes=4)`。旧M1与M2在layer3[5]的conv1+BN+ReLU之后、conv2之前依次作用；M2还包括layer2低层区域池化、layer3深层区域池化及深层1024->512映射，与layer4全局2048拼接形成7170维双极化head输入。

- [bottleneck_evidence.py:12](E:/临时会话/sar_m123_standalone/sar_m123/bottleneck_evidence.py:12)：`RegionPoolWriteback` 的三个区域MLP，C->32->C，末层零初始化；第88至100行明确pre3x3位置。
- [spatial_m12.py:17](E:/临时会话/sar_m123_standalone/sar_m123/current/spatial_m12.py:17)：空间池化自身没有可训练参数；第43至52行去除M1/M2频域，但没有关闭M3频域。
- [selective_backbone.py:23](E:/临时会话/sar_m123_standalone/sar_m123/selective_backbone.py:23)：浅/深池化、deep_output和旧M3组成。
- [final_module_ablations/model.py:125](E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/final_module_ablations_20261029/model.py:125)：旧fit目标 `CE+.1aux+.1rel`。

因此历史phase0不能简写成“只训练M2+head”。虽然最终的早层M1实现绕过旧internal_m1、M3阶段也删掉旧M3对象，它们训练期对局部卷积/M2/head的影响仍被继承。旧M1在最终早层路径未被调用的直接证据：[m1_covariance/run.py:103](E:/临时会话/m1_covariance_20261101/run.py:103)，该路径只保留late internal_m2，另在early层调用新M1。

## 5. M3 aggregate 的输入、权重、参考与端点

[model_task.py:74](E:/临时会话/m3_frequency_increment_20261028/source_hard_task_20261101/oss2/model_task.py:74)：base从已适配载体加载，新增 `MaskedCostLocalMatch(seed,classes=4)`，删掉base.m3，解冻base.head。其余载体包括全部卷积/M1/M2/deep_output冻结；仍需让梯度穿过冻结layer4和深层读出回到matcher。

[prepare_task.py:13](E:/临时会话/m3_frequency_increment_20261028/source_hard_task_20261101/oss2/prepare_task.py:13) 构建新M3缓存：pre_m3是载体在旧M3之前的实时空间特征；desc来自冻结layer1/layer2、coarse_fixed局部描述，20槽位x768；新matcher输出两极化1024x10x10残差。旧载体其他输入频域键在第19行归零，旧M3frequency在base_model关闭。缓存为 `E:/临时会话/m3_frequency_increment_20261028/source_hard_task_20261101/oss2/cache/*.npy`，complete.json记录947条与载体路径。该缓存含旧载体学得的pre_m3/low，**不能用于新载体重训的phase1**；应按新载体生成相同语义输入。

新matcher实现：[scaled_operator.py:19](E:/临时会话/m3_frequency_increment_20261028/local_scale_repair_20261101/scaled_operator.py:19) 继承 [operator_block.py:17](E:/临时会话/m3_frequency_increment_20261028/class_competitive_matching_20261031/operator_block.py:17)。类别输出投影零初始化，残差乘.1；60步Sinkhorn；aggregate用原transport行列边际的外积替换细对应，但仍保留类别条件度量/竞争及参考权重，不是普通均值池化。训练仅使用完整CE，不使用local辅助logits。

困难权重定义：[prepare_task.py:65](E:/临时会话/m3_frequency_increment_20261028/source_hard_task_20261101/oss2/prepare_task.py:65) 先用旧空间原M3载体、固定32参考计算source真类分差，margin<1为hard（112/719，错误66）。每条基础权重为 `1/(本类独立船数 * 本船观测数)`，hard乘2，再按719条的均值归一；训练loss为当前batch加权CE的mean，不是除以batch权重sum。

每批参考抽样：[train_task.py:9](E:/临时会话/m3_frequency_increment_20261028/source_hard_task_20261101/oss2/train_task.py:9)，每类抽8个不同MMSI，再每船抽一条。参考和查询可以来自同一源池，但实际关系排除同MMSI、同platform/orbit、不可用极化；见 [model_task.py:47](E:/临时会话/m3_frequency_increment_20261028/source_hard_task_20261101/oss2/model_task.py:47)。固定验证/推理仍用原32参考。

**注意两阶段排除规则并不相同**：旧载体fit/adapt的 `current.data.pair_mask` 只排同id/MMSI ([data.py:67](E:/临时会话/sar_m123_standalone/sar_m123/current/data.py:67))；aggregate追加platform/orbit排除。新配套实验若统一采用更严格排除，应说明这一协议变化，不能写成逐项原样重放。

M3验证选择：BA优先、OA次之，同值保留早轮，epoch0可选。aggregate_17的 `selected_epoch=0`；当前继承的却是固定epoch4 `last.pt`。curve.csv中epoch4验证OA58.77193/BA50.69179，epoch0为59.21053/50.97269。不要读receipt的selected结果当成last权重结果。新重训若复用工作版末轮制度，应预先写固定120步；若改用验证选优，应标记为新选择制度。

## 6. 数据与实际暴露范围

以下从P/manifests的JSON只读计数，并核对既有 `m1_coverage_20261101/coverage.csv`。类别顺序：散货、普通货、化学品/成品油轮、集装箱；旧名称 `product_chemical_tanker/container_ship` 与当前 `chemical_tanker/container` 是本链中的同一四类映射。

| 集合 | 观测 | 独立MMSI | platform/orbit组 | 四类观测数 |
|---|---:|---:|---:|---|
| source_fit | 478 | 448 | 9 | 138 / 47 / 95 / 198 |
| source_adapt | 241 | 235 | 4 | 83 / 26 / 46 / 86 |
| 合并源池 | 719 | 683 | 13 | 221 / 73 / 141 / 284 |
| validation | 228 | 223 | 4 | 87 / 23 / 29 / 89 |
| selection | 194 | 193 | 5 | 64 / 18 / 40 / 72 |
| evaluation | 378 | 369 | 5 | 132 / 26 / 83 / 137 |
| 固定参考（fit子集） | 32 | 32 | 9 | 8 / 8 / 8 / 8 |

合并源池双极化671条、仅VV16、仅VH32。输入为既有检测后OSS2 fold0、native160 crop/pad，原显示chip uint8/255，VV/VH分开；无resize/C4。此manifest没有可用于报告港口组成的port字段，不根据场景名猜港口。split转换来源为 `oss2_m123_replication_20261026` 的stack/角色清单和reference_bank.csv，仅复用数据组织，**没有加载该目录旧训练权重**；见 [converter.py:77](E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/final_oss2_replication_20261029/converter.py:77)。

至M3末轮的查询呈现次数：478x8 + 241x6 + 719x4 = 8,146，不是8,146条独立数据。adapt的136条选择也只是同一241条中的子集，不新增独立证据。M3阶段使用全部719，validation228用于历史选轮；selection194/evaluation378不进入该M3训练/选样。所选small M1链没有扩成913条，不能把expanded分支曝光误记到所选模型。

## 7. 新存活算子分阶段重训：可复现范围与未决项

主代理已说明新实现顺序：carrier M2+layer3[5]三卷积+head；M3+head；M1+layer2[3].conv3+head；terminal guidance+head，并有full/noM1/noM2/noM3/plain/SE。本文没有审查新代码。依此目标，合理命名是**“共同ImageNet起点、最终存活算子的分阶段配套重训”**，不是历史变结构祖先链的数值完全重放。

可直接固定的最小顺序为：

1. 同份ImageNet与各seed全新存活模块/head；共同参数按同seed匹配。carrier source_fit执行478、8epoch、batch24、160步，卷积3e-5，存活模块/head .001。
2. 对各arm/seed的fit端点，用其真实原图及干预前向重新产生241条source_adapt教师记录；执行6epoch、batch16、96步，只更新layer3[5]三个卷积，保留完整CE/KL/floor目标及系数。
3. 固定适配后载体，重建对应M3输入和困难权重定义；M3+head执行719、4epoch、batch24、120步及本报告三档LR，固定末轮。无M3臂仍给head相同阶段训练机会；参考episode/查询顺序按配对方案记录。
4. 后续M1和terminal阶段采用主代理已核实的准确日程，逐链传递本次权重；不能恢复历史small_class_s17、旧head、旧delta或学得的pre_m3来替代。

**必须在新协议说明、目前尚未由本文核实的新实现选择：**

- 新carrier没有历史旧M1/旧M3，历史fit的 `.1aux_M1+.1relation_M3` 不再有相应输出。carrier只用存活模块可定义的分类目标，是明确的目标变化；不能把它写成原fit目标完全相同，也不能为了照搬系数构造不存在的辅助项。intervention adapt仍是独立必要阶段。
- 历史hard112来自“旧空间原M3载体”的分差。新网络没有该旧算子；必须预先规定由哪个新carrier/共同source-only教师产生hard标记，是否跨臂共享。112不能硬编码，旧source_difficulty.json不能无说明复用。困难标记不同会改变优化目标，需在结果中说明。
- 历史fit/adapt固定32参考，M3训练每批重抽32参考；新干预选样应保持实际预测模型的参考路径。新carrier尚未启用M3时，参考可能不影响它的输出，这不是复现历史teacher数值的保证。
- 新full/noM1/noM2/noM3/plain/SE的教师各自生成；缺失算子的臂仍保留可适用的训练与预算。若使用公共教师/公共hard集合以控制监督条件，这是另外一种可复现协议，必须明说，不能暗中借用历史full训练信号。
- 原载体的旧M3频域训练、旧M1后续被替换、参考排除规则变化以及seed含义，均是历史祖先与存活算子重训的实质区别。上一轮360步统一联合优化未包含这些阶段，不能再称正式分阶段复现。

当前没有发现需要继续追溯更早训练才能确定carrier/adapt/M3超参数的缺口。剩余工作是主代理将完整intervention阶段接入新网络并明确上述新协议选择；不需要继续全历史考古才能开始。

## 8. 本次核对边界

只读了相关代码、配置、JSON/CSV及CPU权重；没有读取整套特征大缓存或运行训练。默认Anaconda的torch导入遇到OpenMP重复库错误后，改用已有 `D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe` 完成CPU核对，没有设置忽略冲突选项、安装依赖或改环境。核对结束CUDA未初始化。本文之外不修改任何文件；Git同步交由主代理统一处理。
